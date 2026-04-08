#include "CommonUtils.h"

#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <spawn.h>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace CommonUtils {
namespace {
bool setCloseOnExec(int fd) {
    int flags = ::fcntl(fd, F_GETFD);
    if (flags < 0) return false;
    return ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == 0;
}

void appendMutableArg(std::vector<std::vector<char>>& storage, const std::string& value) {
    storage.emplace_back(value.begin(), value.end());
    storage.back().push_back('\0');
}
} // namespace

std::string findExecutableInPath(const std::string& command) {
    if (command.empty()) return "";

    const char* pathEnv = std::getenv("PATH");
    if (!pathEnv) return "";

    std::stringstream ss{std::string(pathEnv)};
    std::string token;
    while (std::getline(ss, token, ':')) {
        if (token.empty()) token = ".";
        const std::filesystem::path candidate = std::filesystem::path(token) / command;
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec) && !ec && ::access(candidate.c_str(), X_OK) == 0) {
            return candidate.string();
        }
    }
    return "";
}

bool commandAvailable(const std::string& command) {
    return !findExecutableInPath(command).empty();
}

int spawnProcess(const std::string& executable,
                 const std::vector<std::string>& args,
                 int stdoutFd,
                 int stderrFd) {
    if (executable.empty()) return -1;

    if (stdoutFd >= 0 && !setCloseOnExec(stdoutFd)) return -1;
    if (stderrFd >= 0 && !setCloseOnExec(stderrFd)) return -1;

    posix_spawn_file_actions_t actions;
    if (::posix_spawn_file_actions_init(&actions) != 0) {
        return -1;
    }

    bool actionsReady = true;
    if (stdoutFd >= 0) {
        if (::posix_spawn_file_actions_adddup2(&actions, stdoutFd, STDOUT_FILENO) != 0 ||
            ::posix_spawn_file_actions_addclose(&actions, stdoutFd) != 0) {
            actionsReady = false;
        }
    }
    if (actionsReady && stderrFd >= 0) {
        if (::posix_spawn_file_actions_adddup2(&actions, stderrFd, STDERR_FILENO) != 0 ||
            (stderrFd != stdoutFd && ::posix_spawn_file_actions_addclose(&actions, stderrFd) != 0)) {
            actionsReady = false;
        }
    }

    posix_spawnattr_t attr;
    const bool attrReady = (::posix_spawnattr_init(&attr) == 0);
    if (!actionsReady || !attrReady) {
        if (attrReady) {
            ::posix_spawnattr_destroy(&attr);
        }
        ::posix_spawn_file_actions_destroy(&actions);
        return -1;
    }

    short spawnFlags = 0;
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
    spawnFlags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#endif
    if (spawnFlags != 0 && ::posix_spawnattr_setflags(&attr, spawnFlags) != 0) {
        ::posix_spawnattr_destroy(&attr);
        ::posix_spawn_file_actions_destroy(&actions);
        return -1;
    }

    std::vector<std::vector<char>> argvStorage;
    argvStorage.reserve(args.size() + 1);
    appendMutableArg(argvStorage, executable);
    for (const auto& arg : args) {
        appendMutableArg(argvStorage, arg);
    }

    std::vector<char*> argv;
    argv.reserve(argvStorage.size() + 1);
    for (auto& entry : argvStorage) {
        argv.push_back(entry.data());
    }
    argv.push_back(nullptr);

    pid_t pid = -1;
    const int spawnRc = ::posix_spawn(&pid, executable.c_str(), &actions, &attr, argv.data(), environ);

    ::posix_spawnattr_destroy(&attr);
    ::posix_spawn_file_actions_destroy(&actions);

    if (spawnRc != 0 || pid <= 0) return -1;
    return static_cast<int>(pid);
}

int waitForProcessExitCode(int pid) {
    if (pid <= 0) return -1;

    int status = 0;
    if (::waitpid(static_cast<pid_t>(pid), &status, 0) < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

int spawnProcessAndWait(const std::string& executable,
                        const std::vector<std::string>& args,
                        int stdoutFd,
                        int stderrFd) {
    const int pid = spawnProcess(executable, args, stdoutFd, stderrFd);
    if (pid <= 0) return -1;
    return waitForProcessExitCode(pid);
}

} // namespace CommonUtils
