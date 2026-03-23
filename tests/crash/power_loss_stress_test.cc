#include "pomai/pomai.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <csignal>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

[[noreturn]] void Die(const char* msg) {
  std::cerr << "FAIL: " << msg << "\n";
  std::exit(1);
}

void ChildWriter(const std::string& db_path, std::uint32_t dim) {
  pomai::DBOptions opt;
  opt.path = db_path;
  opt.dim = dim;
  opt.shard_count = 2;
  opt.fsync = pomai::FsyncPolicy::kAlways;

  std::unique_ptr<pomai::DB> db;
  auto st = pomai::DB::Open(opt, &db);
  if (!st.ok()) Die(st.message());

  std::vector<float> v(dim, 0.0f);
  for (std::uint64_t i = 1; i <= 200000; ++i) {
    for (std::uint32_t d = 0; d < dim; ++d) {
      v[d] = static_cast<float>((i + d) % 97) / 97.0f;
    }
    st = db->Put(i, v);
    if (!st.ok()) Die(st.message());
    if ((i % 400) == 0) {
      st = db->Freeze("__default__");
      if (!st.ok()) Die(st.message());
    }
  }
  std::exit(0);
}

void VerifyReopenAndSearch(const std::string& db_path, std::uint32_t dim) {
  pomai::DBOptions opt;
  opt.path = db_path;
  opt.dim = dim;
  opt.shard_count = 2;
  opt.fsync = pomai::FsyncPolicy::kAlways;

  std::unique_ptr<pomai::DB> db;
  auto st = pomai::DB::Open(opt, &db);
  if (!st.ok()) Die("reopen failed");

  std::vector<float> query(dim, 0.1f);
  pomai::SearchResult out;
  st = db->Search(query, 5, &out);
  if (!st.ok()) Die("search after reopen failed");

  st = db->Close();
  if (!st.ok()) Die("close failed");
}

}  // namespace

int main() {
  const std::uint32_t dim = 32;
  const std::string base = "./power_loss_db";
  fs::remove_all(base);
  fs::create_directories(base);

  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> delay_ms(20, 250);

  // 25 rounds of forced power loss simulation.
  for (int round = 0; round < 25; ++round) {
    const std::string db_path = base + "/round_" + std::to_string(round);
    fs::remove_all(db_path);
    fs::create_directories(db_path);

    pid_t pid = fork();
    if (pid < 0) Die("fork failed");
    if (pid == 0) {
      ChildWriter(db_path, dim);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms(rng)));
    (void)kill(pid, SIGKILL);

    int status = 0;
    (void)waitpid(pid, &status, 0);

    VerifyReopenAndSearch(db_path, dim);
  }

  std::cout << "SUCCESS: power-loss stress test passed\n";
  return 0;
}
