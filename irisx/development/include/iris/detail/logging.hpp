namespace iris {
namespace detail {


void log(const std::string& message) {
    bool verbose = true;
    if (verbose) {
  std::cout << message << std::endl;
    }
}

} // namespace detail
} // namespace iris