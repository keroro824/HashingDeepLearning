#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <cassert>


//! convert string to variable of type T. Used to reading floats, int etc from files
template<typename T>
inline T Scan(const std::string &input)
{
  std::stringstream stream(input);
  T ret;
  stream >> ret;
  return ret;
}

//! convert vectors of string to vectors of type T variables
template<typename T>
inline std::vector<T> Scan(const std::vector< std::string > &input)
{
  std::vector<T> output(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    output[i] = Scan<T>(input[i]);
  }
  return output;
}

/** tokenise input string to vector of string. each element has been separated by a character in the delimiters argument.
    The separator can only be 1 character long. The default delimiters are space or tab
*/
inline std::vector<std::string> Tokenize(const std::string& str,
  const std::string& delimiters = " \t")
{
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
template<typename T>
inline std::vector<T> Tokenize(const std::string &input
  , const std::string& delimiters = " \t")
{
  std::vector<std::string> stringVector = Tokenize(input, delimiters);
  return Scan<T>(stringVector);
}

///////////////////////////////////
template<typename T>
class SubVectorConst
{
protected:
  const std::vector<T> *_vecConst;
  size_t _startIdx, _size;

public:
  SubVectorConst()
  : _vecConst(NULL)
  {}

  SubVectorConst(const std::vector<T> &vec, size_t startIdx, size_t size)
    : _vecConst(&vec)
    , _startIdx(startIdx)
    , _size(size)
  {
    assert(_startIdx < _vecConst->size());
    assert(_startIdx + _size <= _vecConst->size());
  }

  virtual ~SubVectorConst() {}

  virtual const T &operator[](size_t idx) const
  { 
    assert(_vecConst);
    assert(idx < _size);
    return (*_vecConst)[_startIdx + idx];
  }

  virtual const T &AAA(size_t idx) const
  {
    return operator[](idx);
  }

  virtual const T *data() const { return _vecConst->data() + _startIdx; }
};

///////////////////////////////////
template<typename T>
class SubVector : public SubVectorConst<T>
{
protected:
  std::vector<T> *_vec;

public:
  SubVector()
    : SubVectorConst<T>()
    , _vec(NULL)
  {}

  SubVector(std::vector<T> &vec, size_t startIdx, size_t size)
    : SubVectorConst<T>(vec, startIdx, size)
    , _vec(&vec)
  {}
  virtual ~SubVector() {}

  virtual const T &operator[](size_t idx) const
  { // shouldn't be needed
    return SubVectorConst<T>::operator[](idx);
  }

  virtual T &operator[](size_t idx)
  {
    assert(_vec);
    assert(idx < SubVectorConst<T>::_size);
    return (*_vec)[SubVectorConst<T>::_startIdx + idx];
  }

  virtual T *data() { return _vec->data() + SubVectorConst<T>::_startIdx; }
};

