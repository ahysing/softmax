#ifndef STDIN_READER_H
#define STDIN_READER_H
#include <memory>
#include <vector>

void putValuesFromStdin(std::weak_ptr<std::vector<double>> valuesPtr, bool verbose);

#endif
