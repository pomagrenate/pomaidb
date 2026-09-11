// util/env.cc — Default and in-memory Env factory implementations.

#include "env.h"
#include "memory_env.h"
#include "posix_env.h"

namespace pomai {

Env* Env::Default() {
  static PosixEnv default_env;
  return &default_env;
}

std::unique_ptr<Env> Env::NewInMemory() {
  return std::make_unique<InMemoryEnv>();
}

}  // namespace pomai
