#include "WebDashboard.h"

#include "AutoConfig.h"
#include "AutomationPipeline.h"
#include "SeldonExceptions.h"

#include <httplib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <netinet/in.h>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <vector>

namespace {
namespace fs = std::filesystem;

std::string nowIsoLike() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return out.str();
}

std::string sanitizeId(std::string value) {
    for (char& c : value) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '-' || c == '_';
        if (!ok) c = '_';
    }
    return value;
}

std::string randomToken(size_t n = 12) {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    static constexpr std::string_view alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::uniform_int_distribution<size_t> dist(0, alphabet.size() - 1);
    std::string out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.push_back(alphabet[dist(rng)]);
    return out;
}

std::string makeId(const std::string& prefix) {
    return prefix + "_" + nowIsoLike() + "_" + randomToken(6);
}

std::string jsonEscape(const std::string& v) {
    std::ostringstream out;
    for (char c : v) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

std::string fileToString(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "";
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool writeText(const fs::path& path, const std::string& body) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << body;
    return static_cast<bool>(out);
}

std::string mimeFromPath(const fs::path& p) {
    const std::string ext = p.extension().string();
    if (ext == ".html") return "text/html";
    if (ext == ".js") return "application/javascript";
    if (ext == ".css") return "text/css";
    if (ext == ".json") return "application/json";
    if (ext == ".png") return "image/png";
    if (ext == ".svg") return "image/svg+xml";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".md") return "text/markdown";
    return "application/octet-stream";
}

std::string shellEscape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    out.push_back('\'');
    for (char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

bool buildAnalysisBundleZip(const fs::path& outputDir,
                           const std::string& analysisId,
                           fs::path& zipPathOut,
                           std::string& errorOut) {
    if (!fs::exists(outputDir) || !fs::is_directory(outputDir)) {
        errorOut = "analysis output directory not found";
        return false;
    }

    std::vector<std::string> relativeFiles;
    const std::array<std::string, 5> reportFiles = {
        "univariate.md",
        "bivariate.md",
        "neural_synthesis.md",
        "final_analysis.md",
        "report.md"
    };
    for (const auto& report : reportFiles) {
        if (fs::exists(outputDir / report) && fs::is_regular_file(outputDir / report)) {
            relativeFiles.push_back(report);
        }
    }

    const fs::path assetsDir = outputDir / "seldon_report_assets";
    if (fs::exists(assetsDir) && fs::is_directory(assetsDir)) {
        for (const auto& entry : fs::recursive_directory_iterator(assetsDir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string ext = entry.path().extension().string();
            if (ext != ".png" && ext != ".svg" && ext != ".jpg" && ext != ".jpeg") {
                continue;
            }
            relativeFiles.push_back(fs::relative(entry.path(), outputDir).string());
        }
    }

    if (relativeFiles.empty()) {
        errorOut = "no report/chart artifacts available for download";
        return false;
    }

    if (std::system("command -v zip >/dev/null 2>&1") != 0) {
        errorOut = "zip utility is not installed on the server host";
        return false;
    }

    zipPathOut = fs::temp_directory_path() /
        ("seldon_" + sanitizeId(analysisId) + "_" + randomToken(8) + ".zip");

    std::ostringstream cmd;
    cmd << "cd " << shellEscape(outputDir.string()) << " && zip -q -r "
        << shellEscape(zipPathOut.string());
    for (const auto& relative : relativeFiles) {
        cmd << " " << shellEscape(relative);
    }

    const int rc = std::system(cmd.str().c_str());
    if (rc != 0 || !fs::exists(zipPathOut) || fs::file_size(zipPathOut) == 0) {
        errorOut = "failed to build zip bundle";
        if (fs::exists(zipPathOut)) {
            fs::remove(zipPathOut);
        }
        return false;
    }

    return true;
}

std::vector<std::vector<std::string>> parseMarkdownTables(const std::string& markdown) {
    std::vector<std::vector<std::string>> rows;
    std::istringstream in(markdown);
    std::string line;
    while (std::getline(in, line)) {
        if (line.find('|') == std::string::npos) continue;
        const std::string trimmed = std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");
        if (trimmed.size() < 3) continue;
        std::vector<std::string> cells;
        std::string cur;
        for (char c : trimmed) {
            if (c == '|') {
                const std::string cell = std::regex_replace(cur, std::regex("^\\s+|\\s+$"), "");
                if (!cell.empty()) cells.push_back(cell);
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
        const std::string tail = std::regex_replace(cur, std::regex("^\\s+|\\s+$"), "");
        if (!tail.empty()) cells.push_back(tail);
        if (cells.empty()) continue;
        bool separator = true;
        for (const auto& c : cells) {
            if (c.find_first_not_of("-:") != std::string::npos) {
                separator = false;
                break;
            }
        }
        if (!separator) rows.push_back(std::move(cells));
    }
    return rows;
}

std::array<uint32_t, 5> sha1(const std::string& input) {
    uint64_t bitLen = static_cast<uint64_t>(input.size()) * 8ULL;
    std::vector<uint8_t> msg(input.begin(), input.end());
    msg.push_back(0x80);
    while ((msg.size() % 64) != 56) msg.push_back(0);
    for (int i = 7; i >= 0; --i) msg.push_back(static_cast<uint8_t>((bitLen >> (i * 8)) & 0xFF));

    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xEFCDAB89;
    uint32_t h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xC3D2E1F0;

    auto rol = [](uint32_t v, int s) { return (v << s) | (v >> (32 - s)); };

    for (size_t chunk = 0; chunk < msg.size(); chunk += 64) {
        uint32_t w[80]{};
        for (int i = 0; i < 16; ++i) {
            const size_t b = chunk + static_cast<size_t>(i) * 4;
            w[i] = (static_cast<uint32_t>(msg[b]) << 24) |
                   (static_cast<uint32_t>(msg[b + 1]) << 16) |
                   (static_cast<uint32_t>(msg[b + 2]) << 8) |
                   (static_cast<uint32_t>(msg[b + 3]));
        }
        for (int i = 16; i < 80; ++i) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        for (int i = 0; i < 80; ++i) {
            uint32_t f = 0;
            uint32_t k = 0;
            if (i < 20) {
                f = (b & c) | ((~b) & d);
                k = 0x5A827999;
            } else if (i < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;
            } else if (i < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }
            const uint32_t temp = rol(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = rol(b, 30);
            b = a;
            a = temp;
        }

        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }

    return {h0, h1, h2, h3, h4};
}

std::string base64Encode(const std::vector<uint8_t>& data) {
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((data.size() * 4 + 2) / 3);
    size_t i = 0;
    while (i + 2 < data.size()) {
        const uint32_t n = (static_cast<uint32_t>(data[i]) << 16) |
                           (static_cast<uint32_t>(data[i + 1]) << 8) |
                           static_cast<uint32_t>(data[i + 2]);
        out.push_back(alphabet[(n >> 18) & 63]);
        out.push_back(alphabet[(n >> 12) & 63]);
        out.push_back(alphabet[(n >> 6) & 63]);
        out.push_back(alphabet[n & 63]);
        i += 3;
    }
    if (i < data.size()) {
        uint32_t n = static_cast<uint32_t>(data[i]) << 16;
        if (i + 1 < data.size()) n |= static_cast<uint32_t>(data[i + 1]) << 8;
        out.push_back(alphabet[(n >> 18) & 63]);
        out.push_back(alphabet[(n >> 12) & 63]);
        if (i + 1 < data.size()) {
            out.push_back(alphabet[(n >> 6) & 63]);
            out.push_back('=');
        } else {
            out.push_back('=');
            out.push_back('=');
        }
    }
    return out;
}

std::string websocketAcceptKey(const std::string& clientKey) {
    static constexpr std::string_view magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    const auto digest = sha1(clientKey + std::string(magic));
    std::vector<uint8_t> bytes;
    bytes.reserve(20);
    for (uint32_t part : digest) {
        bytes.push_back(static_cast<uint8_t>((part >> 24) & 0xFF));
        bytes.push_back(static_cast<uint8_t>((part >> 16) & 0xFF));
        bytes.push_back(static_cast<uint8_t>((part >> 8) & 0xFF));
        bytes.push_back(static_cast<uint8_t>(part & 0xFF));
    }
    return base64Encode(bytes);
}

class ProgressWebSocketHub {
public:
    ~ProgressWebSocketHub() { stop(); }

    bool start(int port) {
        stopRequested.store(false, std::memory_order_relaxed);
        serverFd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (serverFd < 0) return false;

        int opt = 1;
        ::setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        if (::bind(serverFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            ::close(serverFd);
            serverFd = -1;
            return false;
        }
        if (::listen(serverFd, 16) != 0) {
            ::close(serverFd);
            serverFd = -1;
            return false;
        }

        acceptThread = std::thread([this]() { this->acceptLoop(); });
        return true;
    }

    void stop() {
        stopRequested.store(true, std::memory_order_relaxed);
        if (serverFd >= 0) {
            ::shutdown(serverFd, SHUT_RDWR);
            ::close(serverFd);
            serverFd = -1;
        }
        if (acceptThread.joinable()) acceptThread.join();

        std::lock_guard<std::mutex> lock(clientMutex);
        for (int fd : clients) {
            ::shutdown(fd, SHUT_RDWR);
            ::close(fd);
        }
        clients.clear();
    }

    void broadcastText(const std::string& text) {
        std::vector<int> snapshot;
        {
            std::lock_guard<std::mutex> lock(clientMutex);
            snapshot = clients;
        }

        std::vector<int> dead;
        for (int fd : snapshot) {
            if (!sendFrame(fd, text)) dead.push_back(fd);
        }

        if (!dead.empty()) {
            std::lock_guard<std::mutex> lock(clientMutex);
            clients.erase(std::remove_if(clients.begin(), clients.end(), [&](int fd) {
                return std::find(dead.begin(), dead.end(), fd) != dead.end();
            }), clients.end());
            for (int fd : dead) {
                ::shutdown(fd, SHUT_RDWR);
                ::close(fd);
            }
        }
    }

private:
    int serverFd = -1;
    std::thread acceptThread;
    std::atomic<bool> stopRequested{false};
    std::mutex clientMutex;
    std::vector<int> clients;

    static bool sendAll(int fd, const uint8_t* data, size_t len) {
        size_t sent = 0;
        while (sent < len) {
            const ssize_t n = ::send(fd, data + sent, len - sent, 0);
            if (n <= 0) return false;
            sent += static_cast<size_t>(n);
        }
        return true;
    }

    static bool sendFrame(int fd, const std::string& text) {
        std::vector<uint8_t> frame;
        frame.push_back(0x81);
        const size_t n = text.size();
        if (n <= 125) {
            frame.push_back(static_cast<uint8_t>(n));
        } else if (n <= 65535) {
            frame.push_back(126);
            frame.push_back(static_cast<uint8_t>((n >> 8) & 0xFF));
            frame.push_back(static_cast<uint8_t>(n & 0xFF));
        } else {
            frame.push_back(127);
            for (int i = 7; i >= 0; --i) frame.push_back(static_cast<uint8_t>((n >> (i * 8)) & 0xFF));
        }
        frame.insert(frame.end(), text.begin(), text.end());
        return sendAll(fd, frame.data(), frame.size());
    }

    void acceptLoop() {
        while (!stopRequested.load(std::memory_order_relaxed)) {
            sockaddr_in clientAddr{};
            socklen_t len = sizeof(clientAddr);
            const int fd = ::accept(serverFd, reinterpret_cast<sockaddr*>(&clientAddr), &len);
            if (fd < 0) {
                if (stopRequested.load(std::memory_order_relaxed)) break;
                continue;
            }

            if (handshake(fd)) {
                std::lock_guard<std::mutex> lock(clientMutex);
                clients.push_back(fd);
            } else {
                ::close(fd);
            }
        }
    }

    static bool handshake(int fd) {
        std::string req;
        char buf[1024];
        while (req.find("\r\n\r\n") == std::string::npos) {
            const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) return false;
            req.append(buf, buf + n);
            if (req.size() > 16384) return false;
        }

        std::string key;
        std::istringstream in(req);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const std::string lower = std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");
            if (lower.rfind("Sec-WebSocket-Key:", 0) == 0 || lower.rfind("sec-websocket-key:", 0) == 0) {
                const size_t pos = lower.find(':');
                if (pos != std::string::npos) {
                    key = std::regex_replace(lower.substr(pos + 1), std::regex("^\\s+|\\s+$"), "");
                }
            }
        }
        if (key.empty()) return false;

        const std::string accept = websocketAcceptKey(key);
        std::ostringstream resp;
        resp << "HTTP/1.1 101 Switching Protocols\r\n"
             << "Upgrade: websocket\r\n"
             << "Connection: Upgrade\r\n"
             << "Sec-WebSocket-Accept: " << accept << "\r\n\r\n";
        const std::string payload = resp.str();
        return sendAll(fd, reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
    }
};

struct AnalysisInfo {
    std::string id;
    std::string workspaceId;
    std::string name;
    std::string datasetPath;
    std::string target;
    std::string status = "queued";
    std::string message = "Queued";
    int step = 0;
    int totalSteps = 0;
    std::string createdAt;
    std::string startedAt;
    std::string finishedAt;
    std::string outputDir;
    std::string notesPath;
    std::string shareToken;
};

struct WorkspaceInfo {
    std::string id;
    std::string name;
    std::string createdAt;
};

class DashboardBackend {
public:
    explicit DashboardBackend(const WebDashboard::Config& cfg)
        : config(cfg), root(fs::current_path() / ".seldon_web"), wsRoot(root / "workspaces"), shareRoot(root / "shares") {
        fs::create_directories(wsRoot);
        fs::create_directories(shareRoot);
        loadState();
    }

    int run() {
        if (!wsHub.start(config.wsPort)) {
            std::cerr << "[SeldonWeb] failed to start websocket hub on port " << config.wsPort << "\n";
            return 1;
        }

        httplib::Server server;
        server.new_task_queue = [threads = std::max<size_t>(1, config.threads)] {
            return new httplib::ThreadPool(static_cast<int>(threads));
        };
        server.set_default_headers({
            {"X-Content-Type-Options", "nosniff"},
            {"X-Frame-Options", "DENY"},
            {"Referrer-Policy", "strict-origin-when-cross-origin"},
            {"Content-Security-Policy", "default-src 'self'; script-src 'self' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; img-src 'self' data: blob: http: https:; connect-src 'self' ws: wss:; font-src 'self' data:;"}
        });

        wireRoutes(server);

        std::cout << "[SeldonWeb] http://" << config.host << ":" << config.port
                  << " ws://" << config.host << ":" << config.wsPort << "\n";

        if (!server.listen(config.host.c_str(), config.port)) return 1;
        wsHub.stop();
        return 0;
    }

private:
    WebDashboard::Config config;
    fs::path root;
    fs::path wsRoot;
    fs::path shareRoot;

    std::mutex mutex;
    std::unordered_map<std::string, WorkspaceInfo> workspaces;
    std::unordered_map<std::string, AnalysisInfo> analyses;
    std::atomic<bool> pipelineRunning{false};
    ProgressWebSocketHub wsHub;

    void loadState() {
        for (const auto& wsDir : fs::directory_iterator(wsRoot)) {
            if (!wsDir.is_directory()) continue;
            const fs::path meta = wsDir.path() / "workspace.txt";
            if (!fs::exists(meta)) continue;
            WorkspaceInfo ws;
            ws.id = wsDir.path().filename().string();
            ws.name = readKv(meta, "name", ws.id);
            ws.createdAt = readKv(meta, "created_at", "");
            workspaces[ws.id] = ws;

            const fs::path analysesDir = wsDir.path() / "analyses";
            if (!fs::exists(analysesDir)) continue;
            for (const auto& anDir : fs::directory_iterator(analysesDir)) {
                if (!anDir.is_directory()) continue;
                const fs::path ameta = anDir.path() / "analysis.txt";
                if (!fs::exists(ameta)) continue;
                AnalysisInfo info;
                info.id = anDir.path().filename().string();
                info.workspaceId = ws.id;
                info.name = readKv(ameta, "name", info.id);
                info.datasetPath = readKv(ameta, "dataset", "");
                info.target = readKv(ameta, "target", "");
                info.status = readKv(ameta, "status", "unknown");
                info.message = readKv(ameta, "message", "");
                info.step = std::stoi(readKv(ameta, "step", "0"));
                info.totalSteps = std::stoi(readKv(ameta, "total", "0"));
                info.createdAt = readKv(ameta, "created_at", "");
                info.startedAt = readKv(ameta, "started_at", "");
                info.finishedAt = readKv(ameta, "finished_at", "");
                info.outputDir = readKv(ameta, "output_dir", (anDir.path() / "output").string());
                info.notesPath = (anDir.path() / "notes.md").string();
                info.shareToken = readKv(ameta, "share_token", "");
                analyses[info.id] = info;
            }
        }
    }

    static std::string readKv(const fs::path& path, const std::string& key, const std::string& fallback) {
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            const size_t pos = line.find('=');
            if (pos == std::string::npos) continue;
            const std::string k = line.substr(0, pos);
            if (k == key) return line.substr(pos + 1);
        }
        return fallback;
    }

    void persistWorkspace(const WorkspaceInfo& ws) {
        const fs::path dir = wsRoot / ws.id;
        fs::create_directories(dir / "analyses");
        std::ostringstream body;
        body << "id=" << ws.id << "\n";
        body << "name=" << ws.name << "\n";
        body << "created_at=" << ws.createdAt << "\n";
        writeText(dir / "workspace.txt", body.str());
    }

    void persistAnalysis(const AnalysisInfo& a) {
        const fs::path dir = wsRoot / a.workspaceId / "analyses" / a.id;
        fs::create_directories(dir / "output");
        std::ostringstream body;
        body << "id=" << a.id << "\n";
        body << "workspace_id=" << a.workspaceId << "\n";
        body << "name=" << a.name << "\n";
        body << "dataset=" << a.datasetPath << "\n";
        body << "target=" << a.target << "\n";
        body << "status=" << a.status << "\n";
        body << "message=" << a.message << "\n";
        body << "step=" << a.step << "\n";
        body << "total=" << a.totalSteps << "\n";
        body << "created_at=" << a.createdAt << "\n";
        body << "started_at=" << a.startedAt << "\n";
        body << "finished_at=" << a.finishedAt << "\n";
        body << "output_dir=" << a.outputDir << "\n";
        body << "share_token=" << a.shareToken << "\n";
        writeText(dir / "analysis.txt", body.str());
    }

    void wireRoutes(httplib::Server& server) {
        server.Get("/", [&](const httplib::Request&, httplib::Response& res) {
            serveStatic(res, fs::current_path() / "web" / "index.html");
        });
        server.Get("/app.js", [&](const httplib::Request&, httplib::Response& res) {
            serveStatic(res, fs::current_path() / "web" / "app.js");
        });
        server.Get("/api.js", [&](const httplib::Request&, httplib::Response& res) {
            serveStatic(res, fs::current_path() / "web" / "api.js");
        });
        server.Get("/state.js", [&](const httplib::Request&, httplib::Response& res) {
            serveStatic(res, fs::current_path() / "web" / "state.js");
        });
        server.Get("/ui.js", [&](const httplib::Request&, httplib::Response& res) {
            serveStatic(res, fs::current_path() / "web" / "ui.js");
        });
        server.Get("/websocket.js", [&](const httplib::Request&, httplib::Response& res) {
            serveStatic(res, fs::current_path() / "web" / "websocket.js");
        });
        server.Get("/analysis.js", [&](const httplib::Request&, httplib::Response& res) {
            serveStatic(res, fs::current_path() / "web" / "analysis.js");
        });
        server.Get("/analysis.html", [&](const httplib::Request&, httplib::Response& res) {
            serveStatic(res, fs::current_path() / "web" / "analysis.html");
        });
        server.Get(R"(/analysis/([A-Za-z0-9_\-]+))", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string analysisId = req.matches[1];
            std::string html = fileToString(fs::current_path() / "web" / "analysis.html");
            if (html.empty()) {
                res.status = 404;
                return;
            }
            const std::string inject = "<script>window.SELDON_ANALYSIS_ID='" + jsonEscape(analysisId) + "';</script>";
            const size_t pos = html.find("</head>");
            if (pos != std::string::npos) {
                html.insert(pos, inject);
            }
            res.set_content(html, "text/html");
        });

        server.Get("/api/config", [&](const httplib::Request&, httplib::Response& res) {
            std::ostringstream out;
            out << "{\"ws_port\":" << config.wsPort << "}";
            json(res, 200, out.str());
        });

        server.Get("/api/workspaces", [&](const httplib::Request&, httplib::Response& res) {
            std::lock_guard<std::mutex> lock(mutex);
            std::ostringstream out;
            out << "{\"workspaces\":[";
            bool first = true;
            for (const auto& kv : workspaces) {
                if (!first) out << ',';
                first = false;
                out << "{\"id\":\"" << jsonEscape(kv.second.id)
                    << "\",\"name\":\"" << jsonEscape(kv.second.name)
                    << "\",\"created_at\":\"" << jsonEscape(kv.second.createdAt) << "\"}";
            }
            out << "]}";
            json(res, 200, out.str());
        });

        server.Post("/api/workspaces", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string name = req.has_param("name") ? req.get_param_value("name") : "Untitled Workspace";
            WorkspaceInfo ws;
            ws.id = sanitizeId(makeId("ws"));
            ws.name = name;
            ws.createdAt = nowIsoLike();
            {
                std::lock_guard<std::mutex> lock(mutex);
                workspaces[ws.id] = ws;
                persistWorkspace(ws);
            }
            json(res, 200, "{\"id\":\"" + jsonEscape(ws.id) + "\"}");
        });

        server.Delete(R"(/api/workspaces/([A-Za-z0-9_\-]+))", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string wsId = req.matches[1];
            std::lock_guard<std::mutex> lock(mutex);

            auto wsIt = workspaces.find(wsId);
            if (wsIt == workspaces.end()) {
                json(res, 404, "{\"error\":\"workspace not found\"}");
                return;
            }

            for (const auto& kv : analyses) {
                if (kv.second.workspaceId == wsId && kv.second.status == "running") {
                    json(res, 409, "{\"error\":\"cannot delete workspace while analysis is running\"}");
                    return;
                }
            }

            std::vector<std::string> toDelete;
            toDelete.reserve(analyses.size());
            for (const auto& kv : analyses) {
                if (kv.second.workspaceId == wsId) {
                    toDelete.push_back(kv.first);
                }
            }

            for (const auto& analysisId : toDelete) {
                auto it = analyses.find(analysisId);
                if (it == analyses.end()) {
                    continue;
                }
                if (!it->second.shareToken.empty()) {
                    fs::remove(shareRoot / (it->second.shareToken + ".txt"));
                }
                fs::remove_all(wsRoot / wsId / "analyses" / analysisId);
                analyses.erase(it);
            }

            fs::remove_all(wsRoot / wsId);
            workspaces.erase(wsIt);
            json(res, 200, "{\"ok\":true}");
        });

        server.Post("/api/upload", [&](const httplib::Request& req, httplib::Response& res) {
            if (!req.has_param("workspace_id")) {
                json(res, 400, "{\"error\":\"workspace_id is required\"}");
                return;
            }
            const std::string wsId = req.get_param_value("workspace_id");
            if (!req.has_file("dataset")) {
                json(res, 400, "{\"error\":\"multipart file field 'dataset' is required\"}");
                return;
            }
            auto file = req.get_file_value("dataset");
            const fs::path dir = wsRoot / wsId / "uploads";
            fs::create_directories(dir);
            const fs::path path = dir / (sanitizeId(randomToken(8)) + "_" + sanitizeId(file.filename));
            std::ofstream out(path, std::ios::binary);
            out << file.content;
            json(res, 200, "{\"dataset_path\":\"" + jsonEscape(path.string()) + "\"}");
        });

        server.Post(R"(/api/workspaces/([A-Za-z0-9_\-]+)/notes)", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string wsId = req.matches[1];
            const fs::path notesPath = wsRoot / wsId / "workspace_notes.md";
            fs::create_directories(notesPath.parent_path());
            if (!writeText(notesPath, req.body)) {
                json(res, 500, "{\"error\":\"failed to save notes\"}");
                return;
            }
            json(res, 200, "{\"ok\":true}");
        });

        server.Get(R"(/api/workspaces/([A-Za-z0-9_\-]+)/notes)", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string wsId = req.matches[1];
            const fs::path notesPath = wsRoot / wsId / "workspace_notes.md";
            const std::string body = fileToString(notesPath);
            json(res, 200, "{\"markdown\":\"" + jsonEscape(body) + "\"}");
        });

        server.Get(R"(/api/workspaces/([A-Za-z0-9_\-]+)/analyses)", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string wsId = req.matches[1];
            std::lock_guard<std::mutex> lock(mutex);
            std::ostringstream out;
            out << "{\"analyses\":[";
            bool first = true;
            for (const auto& kv : analyses) {
                if (kv.second.workspaceId != wsId) continue;
                if (!first) out << ',';
                first = false;
                appendAnalysisJson(out, kv.second);
            }
            out << "]}";
            json(res, 200, out.str());
        });

        server.Post("/api/analyses/run", [&](const httplib::Request& req, httplib::Response& res) {
            if (!req.has_param("workspace_id") || !req.has_param("dataset_path")) {
                json(res, 400, "{\"error\":\"workspace_id and dataset_path are required\"}");
                return;
            }
            if (pipelineRunning.exchange(true, std::memory_order_acq_rel)) {
                json(res, 409, "{\"error\":\"another analysis is currently running\"}");
                return;
            }

            AnalysisInfo analysis;
            analysis.id = sanitizeId(makeId("analysis"));
            analysis.workspaceId = req.get_param_value("workspace_id");
            analysis.datasetPath = req.get_param_value("dataset_path");
            analysis.target = req.has_param("target") ? req.get_param_value("target") : "";
            analysis.name = req.has_param("name") ? req.get_param_value("name") : analysis.id;
            analysis.createdAt = nowIsoLike();
            analysis.startedAt = nowIsoLike();
            analysis.status = "running";
            analysis.message = "Starting";

            const fs::path analysisDir = wsRoot / analysis.workspaceId / "analyses" / analysis.id;
            analysis.outputDir = (analysisDir / "output").string();
            analysis.notesPath = (analysisDir / "notes.md").string();

            {
                std::lock_guard<std::mutex> lock(mutex);
                analyses[analysis.id] = analysis;
                persistAnalysis(analysis);
            }

            const std::string featureStrategy = req.has_param("feature_strategy") ? req.get_param_value("feature_strategy") : "auto";
            const std::string neuralStrategy = req.has_param("neural_strategy") ? req.get_param_value("neural_strategy") : "auto";
            const std::string bivariateStrategy = req.has_param("bivariate_strategy") ? req.get_param_value("bivariate_strategy") : "auto";
            const std::string plots = req.has_param("plots") ? req.get_param_value("plots") : "bivariate,univariate,overall";
            const std::string plotUnivariate = req.has_param("plot_univariate") ? req.get_param_value("plot_univariate") : "true";
            const std::string plotOverall = req.has_param("plot_overall") ? req.get_param_value("plot_overall") : "true";
            const std::string plotBivariate = req.has_param("plot_bivariate") ? req.get_param_value("plot_bivariate") : "true";
            const std::string benchmarkMode = req.has_param("benchmark_mode") ? req.get_param_value("benchmark_mode") : "true";
            const std::string generateHtml = req.has_param("generate_html") ? req.get_param_value("generate_html") : "false";

            std::thread([=]() {
                runAnalysis(analysis.id,
                            analysis.datasetPath,
                            analysis.outputDir,
                            analysis.target,
                            featureStrategy,
                            neuralStrategy,
                            bivariateStrategy,
                            plots,
                            plotUnivariate,
                            plotOverall,
                            plotBivariate,
                            benchmarkMode,
                            generateHtml);
            }).detach();

            json(res, 200, "{\"analysis_id\":\"" + jsonEscape(analysis.id) + "\"}");
        });

        server.Get(R"(/api/analyses/([A-Za-z0-9_\-]+))", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string id = req.matches[1];
            std::lock_guard<std::mutex> lock(mutex);
            auto it = analyses.find(id);
            if (it == analyses.end()) {
                json(res, 404, "{\"error\":\"analysis not found\"}");
                return;
            }
            std::ostringstream out;
            appendAnalysisJson(out, it->second);
            json(res, 200, out.str());
        });

        server.Get(R"(/api/analyses/([A-Za-z0-9_\-]+)/download)", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string id = req.matches[1];
            AnalysisInfo info;
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto it = analyses.find(id);
                if (it == analyses.end()) {
                    json(res, 404, "{\"error\":\"analysis not found\"}");
                    return;
                }
                info = it->second;
            }

            fs::path zipPath;
            std::string error;
            if (!buildAnalysisBundleZip(info.outputDir, id, zipPath, error)) {
                json(res, 500, "{\"error\":\"" + jsonEscape(error) + "\"}");
                return;
            }

            const std::string payload = fileToString(zipPath);
            fs::remove(zipPath);
            if (payload.empty()) {
                json(res, 500, "{\"error\":\"failed to read generated zip\"}");
                return;
            }

            res.set_header("Content-Disposition", "attachment; filename=\"analysis_" + id + "_bundle.zip\"");
            res.set_content(payload, "application/zip");
        });

        server.Delete(R"(/api/analyses/([A-Za-z0-9_\-]+))", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string id = req.matches[1];
            std::lock_guard<std::mutex> lock(mutex);
            auto it = analyses.find(id);
            if (it == analyses.end()) {
                json(res, 404, "{\"error\":\"analysis not found\"}");
                return;
            }
            if (it->second.status == "running") {
                json(res, 409, "{\"error\":\"cannot delete running analysis\"}");
                return;
            }

            const std::string workspaceId = it->second.workspaceId;
            const std::string shareToken = it->second.shareToken;
            fs::remove_all(wsRoot / workspaceId / "analyses" / id);
            if (!shareToken.empty()) {
                fs::remove(shareRoot / (shareToken + ".txt"));
            }
            analyses.erase(it);
            json(res, 200, "{\"ok\":true}");
        });

        server.Get(R"(/api/analyses/([A-Za-z0-9_\-]+)/results)", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string id = req.matches[1];
            AnalysisInfo info;
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto it = analyses.find(id);
                if (it == analyses.end()) {
                    json(res, 404, "{\"error\":\"analysis not found\"}");
                    return;
                }
                info = it->second;
            }

            const fs::path outDir = info.outputDir;
            const std::string report = fileToString(outDir / "report.md");
            const std::string finalAnalysis = fileToString(outDir / "final_analysis.md");
            const std::string univariate = fileToString(outDir / "univariate.md");
            const std::string bivariate = fileToString(outDir / "bivariate.md");
            std::string neuralSynthesis = fileToString(outDir / "neural_synthesis.md");
            if (neuralSynthesis.empty()) {
                for (const auto& entry : fs::directory_iterator(outDir)) {
                    if (!entry.is_regular_file() || entry.path().extension() != ".md") {
                        continue;
                    }
                    const std::string filename = entry.path().filename().string();
                    if (filename == "report.md" || filename == "final_analysis.md" ||
                        filename == "univariate.md" || filename == "bivariate.md") {
                        continue;
                    }
                    neuralSynthesis = fileToString(entry.path());
                    break;
                }
            }

            std::vector<fs::path> images;
            if (fs::exists(outDir / "seldon_report_assets")) {
                for (const auto& entry : fs::recursive_directory_iterator(outDir / "seldon_report_assets")) {
                    if (!entry.is_regular_file()) continue;
                    const std::string ext = entry.path().extension().string();
                    if (ext == ".png" || ext == ".svg" || ext == ".jpg" || ext == ".jpeg") {
                        images.push_back(entry.path());
                    }
                }
            }
            std::sort(images.begin(), images.end());

            const auto tableRows = parseMarkdownTables(report + "\n" + bivariate + "\n" + univariate);
            const size_t totalRows = tableRows.size();
            size_t offset = 0;
            size_t limit = totalRows;
            if (req.has_param("offset")) {
                try {
                    offset = static_cast<size_t>(std::stoull(req.get_param_value("offset")));
                } catch (...) {
                    offset = 0;
                }
            }
            if (req.has_param("limit")) {
                try {
                    limit = static_cast<size_t>(std::stoull(req.get_param_value("limit")));
                } catch (...) {
                    limit = totalRows;
                }
                if (limit == 0) {
                    limit = totalRows;
                }
            }
            if (offset > totalRows) {
                offset = totalRows;
            }
            const size_t end = std::min(totalRows, offset + limit);
            const bool hasMore = end < totalRows;

            std::vector<std::string> univariateCharts;
            std::vector<std::string> bivariateCharts;
            std::vector<std::string> overallCharts;

            std::ostringstream out;
            out << "{\"report_markdown\":\"" << jsonEscape(report)
                << "\",\"final_markdown\":\"" << jsonEscape(finalAnalysis)
                << "\",\"offset\":" << offset
                << ",\"limit\":" << limit
                << ",\"next_offset\":" << end
                << ",\"total_table_rows\":" << totalRows
                << ",\"has_more\":" << (hasMore ? "true" : "false")
                << ",\"tables\":[";
            for (size_t i = offset; i < end; ++i) {
                if (i > offset) out << ',';
                out << '[';
                for (size_t j = 0; j < tableRows[i].size(); ++j) {
                    if (j > 0) out << ',';
                    out << "\"" << jsonEscape(tableRows[i][j]) << "\"";
                }
                out << ']';
            }
            out << "],\"charts\":[";
            for (size_t i = 0; i < images.size(); ++i) {
                if (i > 0) out << ',';
                const std::string relative = fs::relative(images[i], outDir).string();
                const std::string chartUrl = "/files/" + id + "/" + relative;
                out << "\"" << jsonEscape(chartUrl) << "\"";
                if (relative.find("univariate/") != std::string::npos) {
                    univariateCharts.push_back(chartUrl);
                } else if (relative.find("bivariate/") != std::string::npos) {
                    bivariateCharts.push_back(chartUrl);
                } else {
                    overallCharts.push_back(chartUrl);
                }
            }
            out << "],\"chart_groups\":{"
                << "\"univariate\":[";
            for (size_t i = 0; i < univariateCharts.size(); ++i) {
                if (i > 0) out << ',';
                out << "\"" << jsonEscape(univariateCharts[i]) << "\"";
            }
            out << "],\"bivariate\":[";
            for (size_t i = 0; i < bivariateCharts.size(); ++i) {
                if (i > 0) out << ',';
                out << "\"" << jsonEscape(bivariateCharts[i]) << "\"";
            }
            out << "],\"overall\":[";
            for (size_t i = 0; i < overallCharts.size(); ++i) {
                if (i > 0) out << ',';
                out << "\"" << jsonEscape(overallCharts[i]) << "\"";
            }
            out << "]},\"reports\":{"
                << "\"univariate\":\"" << jsonEscape(univariate) << "\"," 
                << "\"bivariate\":\"" << jsonEscape(bivariate) << "\"," 
                << "\"neural_synthesis\":\"" << jsonEscape(neuralSynthesis) << "\"," 
                << "\"final_analysis\":\"" << jsonEscape(finalAnalysis) << "\"," 
                << "\"report\":\"" << jsonEscape(report) << "\"}}";

            json(res, 200, out.str());
        });

        server.Get(R"(/files/([A-Za-z0-9_\-]+)/(.+))", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string id = req.matches[1];
            const std::string rel = req.matches[2];
            AnalysisInfo info;
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto it = analyses.find(id);
                if (it == analyses.end()) {
                    res.status = 404;
                    return;
                }
                info = it->second;
            }
            fs::path requested = fs::weakly_canonical(fs::path(info.outputDir) / rel);
            const fs::path base = fs::weakly_canonical(fs::path(info.outputDir));
            if (requested.string().rfind(base.string(), 0) != 0 || !fs::exists(requested)) {
                res.status = 404;
                return;
            }
            serveStatic(res, requested);
        });

        server.Post(R"(/api/analyses/([A-Za-z0-9_\-]+)/notes)", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string id = req.matches[1];
            AnalysisInfo info;
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto it = analyses.find(id);
                if (it == analyses.end()) {
                    json(res, 404, "{\"error\":\"analysis not found\"}");
                    return;
                }
                info = it->second;
            }
            if (!writeText(info.notesPath, req.body)) {
                json(res, 500, "{\"error\":\"failed to save notes\"}");
                return;
            }
            json(res, 200, "{\"ok\":true}");
        });

        server.Get(R"(/api/analyses/([A-Za-z0-9_\-]+)/notes)", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string id = req.matches[1];
            AnalysisInfo info;
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto it = analyses.find(id);
                if (it == analyses.end()) {
                    json(res, 404, "{\"error\":\"analysis not found\"}");
                    return;
                }
                info = it->second;
            }
            json(res, 200, "{\"markdown\":\"" + jsonEscape(fileToString(info.notesPath)) + "\"}");
        });

        server.Post(R"(/api/analyses/([A-Za-z0-9_\-]+)/share)", [&](const httplib::Request& req, httplib::Response& res) {
            (void)req;
            const std::string id = req.matches[1];
            std::lock_guard<std::mutex> lock(mutex);
            auto it = analyses.find(id);
            if (it == analyses.end()) {
                json(res, 404, "{\"error\":\"analysis not found\"}");
                return;
            }
            if (it->second.shareToken.empty()) {
                it->second.shareToken = randomToken(18);
                writeText(shareRoot / (it->second.shareToken + ".txt"), it->second.id);
                persistAnalysis(it->second);
            }
            const std::string link = "/share/" + it->second.shareToken;
            json(res, 200, "{\"link\":\"" + jsonEscape(link) + "\"}");
        });

        server.Get(R"(/api/share/([A-Za-z0-9]+))", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string token = req.matches[1];
            const fs::path mapFile = shareRoot / (token + ".txt");
            if (!fs::exists(mapFile)) {
                json(res, 404, "{\"error\":\"share not found\"}");
                return;
            }
            const std::string analysisId = fileToString(mapFile);
            AnalysisInfo info;
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto it = analyses.find(analysisId);
                if (it == analyses.end()) {
                    json(res, 404, "{\"error\":\"analysis not found\"}");
                    return;
                }
                info = it->second;
            }
            const std::string notes = fileToString(info.notesPath);
            const std::string summary = fileToString(fs::path(info.outputDir) / "final_analysis.md");
            json(res, 200,
                 "{\"analysis_id\":\"" + jsonEscape(info.id) +
                 "\",\"name\":\"" + jsonEscape(info.name) +
                 "\",\"status\":\"" + jsonEscape(info.status) +
                 "\",\"notes\":\"" + jsonEscape(notes) +
                 "\",\"summary\":\"" + jsonEscape(summary) + "\"}");
        });

        server.Get(R"(/share/([A-Za-z0-9]+))", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string token = req.matches[1];
            const fs::path htmlPath = fs::current_path() / "web" / "index.html";
            std::string html = fileToString(htmlPath);
            if (html.empty()) {
                res.status = 404;
                return;
            }
            const std::string inject = "<script>window.SELDON_SHARE_TOKEN='" + jsonEscape(token) + "';</script>";
            const size_t pos = html.find("</head>");
            if (pos != std::string::npos) html.insert(pos, inject);
            res.set_content(html, "text/html");
        });
    }

    static void json(httplib::Response& res, int code, const std::string& payload) {
        res.status = code;
        res.set_content(payload, "application/json");
    }

    static void serveStatic(httplib::Response& res, const fs::path& path) {
        const std::string body = fileToString(path);
        if (body.empty() && !fs::exists(path)) {
            res.status = 404;
            return;
        }
        res.set_content(body, mimeFromPath(path));
    }

    static void appendAnalysisJson(std::ostringstream& out, const AnalysisInfo& a) {
        out << "{\"id\":\"" << jsonEscape(a.id)
            << "\",\"workspace_id\":\"" << jsonEscape(a.workspaceId)
            << "\",\"name\":\"" << jsonEscape(a.name)
            << "\",\"dataset_path\":\"" << jsonEscape(a.datasetPath)
            << "\",\"target\":\"" << jsonEscape(a.target)
            << "\",\"status\":\"" << jsonEscape(a.status)
            << "\",\"message\":\"" << jsonEscape(a.message)
            << "\",\"step\":" << a.step
            << ",\"total_steps\":" << a.totalSteps
            << ",\"created_at\":\"" << jsonEscape(a.createdAt)
            << "\",\"started_at\":\"" << jsonEscape(a.startedAt)
            << "\",\"finished_at\":\"" << jsonEscape(a.finishedAt)
            << "\",\"output_dir\":\"" << jsonEscape(a.outputDir)
            << "\"}";
    }

    void publishProgress(const AnalysisInfo& a) {
        std::ostringstream out;
        out << "{\"type\":\"progress\",\"analysis_id\":\"" << jsonEscape(a.id)
            << "\",\"status\":\"" << jsonEscape(a.status)
            << "\",\"message\":\"" << jsonEscape(a.message)
            << "\",\"step\":" << a.step
            << ",\"total_steps\":" << a.totalSteps
            << "}";
        wsHub.broadcastText(out.str());
    }

    void runAnalysis(const std::string& analysisId,
                     const std::string& datasetPath,
                     const std::string& outputDir,
                     const std::string& target,
                     const std::string& featureStrategy,
                     const std::string& neuralStrategy,
                     const std::string& bivariateStrategy,
                     const std::string& plots,
                     const std::string& plotUnivariate,
                     const std::string& plotOverall,
                     const std::string& plotBivariate,
                     const std::string& benchmarkMode,
                     const std::string& generateHtml) {
        auto updateState = [&](const std::string& status,
                               const std::string& message,
                               int step,
                               int total) {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = analyses.find(analysisId);
            if (it == analyses.end()) return;
            it->second.status = status;
            it->second.message = message;
            it->second.step = step;
            it->second.totalSteps = total;
            persistAnalysis(it->second);
            publishProgress(it->second);
        };

        try {
            std::vector<std::string> args = {
                "seldon",
                datasetPath,
                "--output-dir", outputDir,
                "--feature-strategy", featureStrategy,
                "--neural-strategy", neuralStrategy,
                "--bivariate-strategy", bivariateStrategy,
                "--plots", plots,
                "--plot-univariate", plotUnivariate,
                "--plot-overall", plotOverall,
                "--plot-bivariate", plotBivariate,
                "--benchmark-mode", benchmarkMode,
                "--generate-html", generateHtml,
                "--verbose-analysis", "false"
            };
            if (!target.empty()) {
                args.push_back("--target");
                args.push_back(target);
            }

            std::vector<char*> argv;
            argv.reserve(args.size());
            for (auto& arg : args) argv.push_back(arg.data());

            AutomationPipeline::onProgress = [&](const std::string& label, int step, int total) {
                updateState("running", label, step, total);
            };

            updateState("running", "Preparing", 0, 10);

            AutoConfig cfg = AutoConfig::fromArgs(static_cast<int>(argv.size()), argv.data());
            AutomationPipeline pipeline;
            const int code = pipeline.run(cfg);

            if (code == 0) {
                std::lock_guard<std::mutex> lock(mutex);
                auto it = analyses.find(analysisId);
                if (it != analyses.end()) {
                    it->second.status = "completed";
                    it->second.message = "Completed";
                    it->second.step = it->second.totalSteps;
                    it->second.finishedAt = nowIsoLike();
                    persistAnalysis(it->second);
                    publishProgress(it->second);
                }
            } else {
                updateState("failed", "Pipeline exited with code " + std::to_string(code), 0, 0);
            }
        } catch (const std::exception& e) {
            updateState("failed", e.what(), 0, 0);
        }

        AutomationPipeline::onProgress = nullptr;
        pipelineRunning.store(false, std::memory_order_release);
    }
};
} // namespace

int WebDashboard::start(const Config& config) {
    DashboardBackend backend(config);
    return backend.run();
}
