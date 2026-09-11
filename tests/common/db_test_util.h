#pragma once
#include <memory>
#include <string>

#include "options.h"
#include "pomai.h"
#include "status.h"

namespace pomai::test
{

    inline std::unique_ptr<pomai::DB> OpenDB(const pomai::DBOptions &opt)
    {
        std::unique_ptr<pomai::DB> db;
        auto st = pomai::DB::Open(opt, &db);
        if (!st.ok())
            return nullptr;
        return db;
    }

} // namespace pomai::test
