#ifndef SELDON_EXCEPTIONS_H
#define SELDON_EXCEPTIONS_H

#include <stdexcept>
#include <string>

namespace Seldon {

class SeldonException : public std::runtime_error {
public:
    explicit SeldonException(const std::string& message) : std::runtime_error(message) {}
};

class IOException : public SeldonException {
public:
    explicit IOException(const std::string& message) : SeldonException("IO Error: " + message) {}
};

class DatasetException : public SeldonException {
public:
    explicit DatasetException(const std::string& message) : SeldonException("Dataset Error: " + message) {}
};

class NeuralNetException : public SeldonException {
public:
    explicit NeuralNetException(const std::string& message) : SeldonException("NeuralNet Error: " + message) {}
};

} // namespace Seldon

#endif // SELDON_EXCEPTIONS_H
