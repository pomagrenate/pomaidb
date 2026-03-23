namespace ruy { class Context {}; class ThreadPool {}; class ScopedSuppressDenormals { public: ScopedSuppressDenormals() {} ~ScopedSuppressDenormals() {} }; } 
namespace ruy { enum class Side { kLhs, kRhs }; } 
