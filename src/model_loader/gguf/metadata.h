#pragma once
#include <cstdint>
#include <string_view>
#include <string>
#include <vector>
#include <memory>
#include <variant>

enum MetadataDType : uint32_t
{
  // The value is a 8-bit unsigned integer.
  GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
  // The value is a 8-bit signed integer.
  GGUF_METADATA_VALUE_TYPE_INT8 = 1,
  // The value is a 16-bit unsigned little-endian integer.
  GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
  // The value is a 16-bit signed little-endian integer.
  GGUF_METADATA_VALUE_TYPE_INT16 = 3,
  // The value is a 32-bit unsigned little-endian integer.
  GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
  // The value is a 32-bit signed little-endian integer.
  GGUF_METADATA_VALUE_TYPE_INT32 = 5,
  // The value is a 32-bit IEEE754 floating point number.
  GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
  // The value is a boolean.
  // 1-byte value where 0 is false and 1 is true.
  // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
  GGUF_METADATA_VALUE_TYPE_BOOL = 7,
  // The value is a UTF-8 non-null-terminated string, with length prepended.
  GGUF_METADATA_VALUE_TYPE_STRING = 8,
  // The value is an array of other values, with the length and type prepended.
  ///
  // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
  GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
  // The value is a 64-bit unsigned little-endian integer.
  GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
  // The value is a 64-bit signed little-endian integer.
  GGUF_METADATA_VALUE_TYPE_INT64 = 11,
  // The value is a 64-bit IEEE754 floating point number.
  GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

constexpr std::string_view metadata_dtype_name(MetadataDType t)
{
  switch (t)
  {
  case GGUF_METADATA_VALUE_TYPE_UINT8:
    return "UINT8";
  case GGUF_METADATA_VALUE_TYPE_INT8:
    return "INT8";
  case GGUF_METADATA_VALUE_TYPE_UINT16:
    return "UINT16";
  case GGUF_METADATA_VALUE_TYPE_INT16:
    return "INT16";
  case GGUF_METADATA_VALUE_TYPE_UINT32:
    return "UINT32";
  case GGUF_METADATA_VALUE_TYPE_INT32:
    return "INT32";
  case GGUF_METADATA_VALUE_TYPE_FLOAT32:
    return "FLOAT32";
  case GGUF_METADATA_VALUE_TYPE_BOOL:
    return "BOOL";
  case GGUF_METADATA_VALUE_TYPE_STRING:
    return "STRING";
  case GGUF_METADATA_VALUE_TYPE_ARRAY:
    return "ARRAY";
  case GGUF_METADATA_VALUE_TYPE_UINT64:
    return "UINT64";
  case GGUF_METADATA_VALUE_TYPE_INT64:
    return "INT64";
  case GGUF_METADATA_VALUE_TYPE_FLOAT64:
    return "FLOAT64";
  }
  return "UNKNOWN";
};

using primitive_value_t = std::variant<
    uint8_t,  // index 0
    int8_t,   // index 1
    uint16_t, // index 2
    int16_t,  // index 3
    uint32_t, // index 4
    int32_t,  // index 5
    float,    // index 6
    bool,     // index 7
    uint64_t, // index 8
    int64_t,  // index 9
    double    // index 10
    >;

class MetadataValue
{
public:
  virtual ~MetadataValue() = default;
  virtual int size() const = 0;
  virtual std::string to_string() const = 0;
  static std::unique_ptr<MetadataValue> from_ptr(const char *ptr, MetadataDType dtype);
};

class MetadataPrimitiveValue : public MetadataValue
{
private:
  primitive_value_t data_;
  int size_;

public:
  MetadataPrimitiveValue(primitive_value_t data, int s) : data_(data), size_(s) {}
  int size() const override;
  std::string to_string() const override;
  primitive_value_t value() const;
  static MetadataPrimitiveValue from_ptr(const char *ptr, MetadataDType dtype);
};

class MetadataStringValue : public MetadataValue
{
private:
  const uint64_t *len_;
  std::string_view data_;
  uint64_t offset_;

public:
  MetadataStringValue(std::string_view s, const uint64_t *l) : data_(s), len_(l), offset_(0) {}
  int size() const override;
  std::string to_string() const override;
  static MetadataStringValue from_ptr(const char *ptr);
};

class MetadataArrayValue : public MetadataValue
{
private:
  const MetadataDType *val_type_;
  const uint64_t *len_;
  std::vector<std::unique_ptr<MetadataValue>> elements_;

  MetadataArrayValue(const MetadataDType *vt, const uint64_t *l) : val_type_(std::move(vt)), len_(std::move(l)) {}

public:
  static MetadataArrayValue from_ptr(const char *ptr);
  std::string to_string() const override;
  const std::vector<std::unique_ptr<MetadataValue>> &elements() const;
  int size() const override;
};

class Metadata
{
private:
  MetadataStringValue key_;
  MetadataDType val_type_;
  std::unique_ptr<MetadataValue> val_;

  Metadata(MetadataStringValue k, MetadataDType vt, std::unique_ptr<MetadataValue> v) : key_(k), val_type_(vt), val_(std::move(v)) {}

public:
  static Metadata from_ptr(const char *ptr);
  int size() const;
  std::string to_string() const;
  std::string key_string() const { return key_.to_string(); }
  MetadataDType dtype() const { return val_type_; }
  std::string value_string() const { return val_->to_string(); }
  const MetadataValue *value() const { return val_.get(); }
};
