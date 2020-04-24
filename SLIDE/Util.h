#pragma once
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

//! convert string to variable of type T. Used to reading floats, int etc from
//! files
template <typename T> inline T Scan(const std::string &input) {
  std::stringstream stream(input);
  T ret;
  stream >> ret;
  return ret;
}

//! convert vectors of string to vectors of type T variables
template <typename T>
inline std::vector<T> Scan(const std::vector<std::string> &input) {
  std::vector<T> output(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    output[i] = Scan<T>(input[i]);
  }
  return output;
}

/** tokenise input string to vector of string. each element has been separated
   by a character in the delimiters argument. The separator can only be 1
   character long. The default delimiters are space or tab
*/
inline std::vector<std::string>
Tokenize(const std::string &str, const std::string &delimiters = " \t") {
  std::vector<std::string> tokens;
  // Skip delimiters at beginning.
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos) {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }

  return tokens;
}

//! tokenise input string to vector of type T
template <typename T>
inline std::vector<T> Tokenize(const std::string &input,
                               const std::string &delimiters = " \t") {
  std::vector<std::string> stringVector = Tokenize(input, delimiters);
  return Scan<T>(stringVector);
}

////////////////////////////////////////////////////////
template <typename T> using Vec2d = std::vector<std::vector<T>>;

template <typename T> using Vec3d = std::vector<std::vector<std::vector<T>>>;

////////////////////////////////////////////////////////
template <typename T> class SubVectorConst {
protected:
  const T *_ptrConst;
  size_t _size;

public:
  SubVectorConst() {}

  SubVectorConst(const std::vector<T> &vec, size_t startIdx, size_t size)
      : _ptrConst(vec.data() + startIdx), _size(size) {
    assert(startIdx < vec.size());
    assert(startIdx + size <= vec.size());
  }

  virtual const T &operator[](size_t idx) const {
    assert(_ptrConst);
    assert(idx < SubVectorConst<T>::_size);
    return _ptrConst[idx];
  }

  size_t size() const { return _size; }
  const T *data() const { return _ptrConst; }
};

////////////////////////////////////////////////////////
template <typename T> class SubVector : public SubVectorConst<T> {
protected:
  T *_ptr;

public:
  SubVector() : SubVectorConst<T>() {}

  SubVector(std::vector<T> &vec, size_t startIdx, size_t size)
      : SubVectorConst<T>(vec, startIdx, size), _ptr(vec.data() + startIdx) {}

  const T &operator[](size_t idx) const { // shouldn't need this
    return SubVectorConst<T>::operator[](idx);
  }

  virtual T &operator[](size_t idx) {
    assert(_ptr);
    assert(idx < SubVectorConst<T>::_size);
    return _ptr[idx];
  }

  T *data() { return _ptr; }
};

////////////////////////////////////////////////////////
template <typename T>
void Print(const std::string &str, const std::vector<T> &vec) {
  SubVectorConst<T> subvec(vec, 0, vec.size());
  Print(str, subvec);
}

template <typename T>
void Print(const std::string &str, const SubVectorConst<T> &vec) {
  std::cerr << str << " " << vec.size() << "=" << std::flush;
  for (size_t i = 0; i < vec.size(); ++i) {
    std::cerr << vec[i] << " "; // << std::endl;
  }
  std::cerr << std::endl;
}

template <typename T>
void PrintSizes(const std::string &str, const Vec2d<T> &vec) {
  std::cerr << str << " " << vec.size() << "=" << std::flush;
  for (size_t i = 0; i < vec.size(); ++i) {
    const std::vector<T> &inner = vec[i];
    std::cerr << inner.size() << " "; // << std::endl;
  }
  std::cerr << std::endl;
}

void CreateData(std::ifstream &file, Vec2d<float> &data, Vec2d<int> &labels,
                int batchSize, size_t inputDim);

void CreateData(std::ifstream &file, Vec2d<int> &records, Vec2d<float> &values,
                Vec2d<int> &labels, int batchsize);
