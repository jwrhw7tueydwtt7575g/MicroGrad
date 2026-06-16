#include "micrograd/serializer.hpp"
#include "micrograd/adam.hpp"
#include "micrograd/sgd.hpp"
#include "micrograd/momentum.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <cstdio>
#include <cstring>

namespace micrograd {

namespace {

// Simple FNV-1a hash for content addressing.
std::string fnv1a(const uint8_t* data, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return std::string(buf);
}

std::string shape_to_str(const Shape& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.dims.size(); ++i) {
        if (i) r += ",";
        r += std::to_string(s.dims[i]);
    }
    r += "]";
    return r;
}

}  // namespace

std::string Serializer::save_blob(const Tensor& t, const std::string& dir) {
    std::filesystem::create_directories(dir);
    const float* p = static_cast<const float*>(t.data_ptr());
    int64_t n = t.numel();
    std::string h = fnv1a(reinterpret_cast<const uint8_t*>(p), static_cast<size_t>(n) * sizeof(float));
    std::string path = dir + "/" + h + ".bin";
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_blob: cannot open " + path);
    f.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n) * sizeof(float));
    return h;
}

Tensor Serializer::load_blob(const std::string& dir, const std::string& hash,
                             Shape shape, DType dtype, Device device) {
    Tensor t = Tensor::empty(shape, dtype, device);
    std::string path = dir + "/" + hash + ".bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_blob: cannot open " + path);
    int64_t n = shape.numel();
    f.read(static_cast<char*>(t.data_ptr()), static_cast<std::streamsize>(n) * sizeof(float));
    return t;
}

void Serializer::save_function(const Function& f, const std::string& path) {
    const Graph& g = f.graph();
    std::ofstream out(path);
    if (!out) throw std::runtime_error("save_function: cannot open " + path);
    out << "{\n  \"version\": 1,\n  \"output_id\": " << g.output_id << ",\n";
    out << "  \"input_ids\": [";
    for (size_t i = 0; i < g.input_ids.size(); ++i) { if (i) out << ","; out << g.input_ids[i]; }
    out << "],\n  \"param_ids\": [";
    for (size_t i = 0; i < g.param_ids.size(); ++i) { if (i) out << ","; out << g.param_ids[i]; }
    out << "],\n  \"nodes\": [\n";
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        const auto& n = g.nodes[i];
        out << "    {\"id\":" << n.id
            << ",\"op\":\"" << n.op_name << "\""
            << ",\"dtype\":" << static_cast<int>(n.dtype)
            << ",\"device_type\":" << static_cast<int>(n.device.type)
            << ",\"device_index\":" << n.device.index
            << ",\"inputs\":[";
        for (size_t k = 0; k < n.inputs.size(); ++k) { if (k) out << ","; out << n.inputs[k]; }
        out << "],\"input_shapes\":[";
        for (size_t k = 0; k < n.cached_inputs.size(); ++k) {
            if (k) out << ",";
            out << "\"" << shape_to_str(n.cached_inputs[k].shape()) << "\"";
        }
        out << "],\"output_shapes\":[";
        for (size_t k = 0; k < n.cached_outputs.size(); ++k) {
            if (k) out << ",";
            out << "\"" << shape_to_str(n.cached_outputs[k].shape()) << "\"";
        }
        out << "]}";
        if (i + 1 < g.nodes.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n}\n";
}

FunctionPtr Serializer::load_function(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("load_function: cannot open " + path);
    std::stringstream ss; ss << in.rdbuf();
    std::string s = ss.str();
    // Minimal JSON parse for the fields we wrote. v0.1 only round-trips the
    // graph structure (shapes/op names), not tensor data.
    Graph g;
    auto find_int = [&](const std::string& key) -> int {
        auto kp = s.find("\"" + key + "\":");
        if (kp == std::string::npos) return -1;
        kp += key.size() + 3;
        while (kp < s.size() && (s[kp] == ' ' || s[kp] == '\n')) kp++;
        int v = 0; bool neg = false;
        if (s[kp] == '-') { neg = true; kp++; }
        while (kp < s.size() && std::isdigit((unsigned char)s[kp])) {
            v = v * 10 + (s[kp] - '0'); kp++;
        }
        return neg ? -v : v;
    };
    g.output_id = find_int("output_id");
    // Re-create nodes in stored order; we re-record shape metadata.
    // Locate the "nodes" array.
    auto np = s.find("\"nodes\":");
    if (np == std::string::npos) throw std::runtime_error("load_function: no nodes");
    auto arr_start = s.find('[', np);
    auto arr_end = s.rfind(']');
    std::string body = s.substr(arr_start, arr_end - arr_start + 1);
    int pos = 0;
    while (true) {
        auto ob_start = body.find('{', pos);
        if (ob_start == std::string::npos) break;
        auto ob_end = body.find('}', ob_start);
        std::string obj = body.substr(ob_start, ob_end - ob_start + 1);
        int id = -1;
        {
            auto kp = obj.find("\"id\":");
            if (kp != std::string::npos) {
                kp += 5;
                while (kp < obj.size() && (obj[kp] == ' ' || obj[kp] == '\n')) kp++;
                int v = 0; while (kp < obj.size() && std::isdigit((unsigned char)obj[kp])) { v = v*10 + (obj[kp]-'0'); kp++; }
                id = v;
            }
        }
        std::string opn;
        {
            auto kp = obj.find("\"op\":\"");
            if (kp != std::string::npos) {
                kp += 6;
                while (kp < obj.size() && obj[kp] != '"') { opn.push_back(obj[kp]); kp++; }
            }
        }
        int new_id = g.add_node();
        IRNode& n = g.node(new_id);
        n.op_name = opn;
        n.id = new_id;
        // Parse inputs.
        auto kp = obj.find("\"inputs\":[");
        if (kp != std::string::npos) {
            kp += 10;
            while (kp < obj.size() && obj[kp] != ']') {
                while (kp < obj.size() && (obj[kp] == ' ' || obj[kp] == ',' || obj[kp] == '\n')) kp++;
                int v = 0; bool neg = false;
                if (obj[kp] == '-') { neg = true; kp++; }
                while (kp < obj.size() && std::isdigit((unsigned char)obj[kp])) { v = v*10 + (obj[kp]-'0'); kp++; }
                n.inputs.push_back(neg ? -v : v);
            }
        }
        (void)id;
        pos = ob_end + 1;
    }
    return std::make_shared<Function>(std::move(g));
}

void Serializer::save_optimizer(const Optimizer& opt, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("save_optimizer: cannot open " + path);
    out << "{\"version\":1,\"lr\":" << opt.lr() << "}\n";
}

void Serializer::load_optimizer(Optimizer& opt, const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("load_optimizer: cannot open " + path);
    std::stringstream ss; ss << in.rdbuf();
    std::string s = ss.str();
    auto kp = s.find("\"lr\":");
    if (kp == std::string::npos) return;
    kp += 5;
    while (kp < s.size() && (s[kp] == ' ' || s[kp] == '\n')) kp++;
    float v = 0.0f;
    sscanf(s.c_str() + kp, "%f", &v);
    opt.set_lr(v);
}

}  // namespace micrograd
