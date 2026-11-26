// json_mini.h - pembaca JSON kecil (RFC 8259) untuk manifest.json. Hanya membaca;
// penulisan JSON dilakukan dengan printf di landcover.cpp (sengaja tanpa pustaka JSON).
//
// Ketat: tanda koma sisa, komentar, NaN/Infinity, dan karakter kontrol mentah di dalam teks
// ditolak. BOM UTF-8 di awal diabaikan (mis. berkas dari PowerShell). Kedalaman maksimum 64.
#ifndef LANDCOVER_JSON_MINI_H
#define LANDCOVER_JSON_MINI_H

#include <cerrno>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace lc {

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    bool is_integer = false;  // angka tanpa pecahan/eksponen yang muat di long long
    long long integer = 0;
    std::string string;
    std::vector<Json> array;
    std::vector<std::pair<std::string, Json>> object;

    // Anggota objek pertama bernama `key`, atau nullptr.
    const Json* find(const std::string& key) const {
        if (type != Type::Object) return nullptr;
        for (const auto& kv : object)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

namespace detail {

class JsonParser {
public:
    explicit JsonParser(const std::string& t) : s_(t) {}

    bool parse(Json& out, std::string& err) {
        if (s_.compare(0, 3, "\xEF\xBB\xBF") == 0) p_ = 3;
        skip_ws();
        if (!value(out, 0)) { err = err_; return false; }
        skip_ws();
        if (p_ != s_.size()) { err = fail("sisa teks setelah nilai JSON"); return false; }
        return true;
    }

private:
    const std::string& s_;
    size_t p_ = 0;
    std::string err_;

    std::string fail(const std::string& what) {
        err_ = what + " (posisi " + std::to_string(p_) + ")";
        return err_;
    }
    bool bad(const std::string& what) { fail(what); return false; }

    void skip_ws() {
        while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\n' || s_[p_] == '\r'))
            ++p_;
    }

    bool literal(const char* word) {
        size_t n = 0;
        while (word[n]) ++n;
        if (s_.compare(p_, n, word) != 0) return false;
        p_ += n;
        return true;
    }

    bool value(Json& out, int depth) {
        if (depth > 64) return bad("JSON terlalu dalam");
        if (p_ >= s_.size()) return bad("JSON berakhir terlalu dini");
        const char c = s_[p_];
        if (c == '{') return object(out, depth);
        if (c == '[') return array(out, depth);
        if (c == '"') { out.type = Json::Type::String; return string(out.string); }
        if (c == 't') { if (!literal("true")) return bad("nilai tidak dikenal"); out.type = Json::Type::Bool; out.boolean = true; return true; }
        if (c == 'f') { if (!literal("false")) return bad("nilai tidak dikenal"); out.type = Json::Type::Bool; out.boolean = false; return true; }
        if (c == 'n') { if (!literal("null")) return bad("nilai tidak dikenal"); out.type = Json::Type::Null; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) return number(out);
        return bad(std::string("karakter tak terduga '") + c + "'");
    }

    bool object(Json& out, int depth) {
        out.type = Json::Type::Object;
        ++p_;  // '{'
        skip_ws();
        if (p_ < s_.size() && s_[p_] == '}') { ++p_; return true; }
        for (;;) {
            skip_ws();
            if (p_ >= s_.size() || s_[p_] != '"') return bad("kunci objek harus teks");
            std::string key;
            if (!string(key)) return false;
            skip_ws();
            if (p_ >= s_.size() || s_[p_] != ':') return bad("':' diharapkan setelah kunci");
            ++p_;
            skip_ws();
            Json v;
            if (!value(v, depth + 1)) return false;
            out.object.emplace_back(std::move(key), std::move(v));
            skip_ws();
            if (p_ >= s_.size()) return bad("objek tidak ditutup");
            if (s_[p_] == ',') { ++p_; continue; }
            if (s_[p_] == '}') { ++p_; return true; }
            return bad("',' atau '}' diharapkan");
        }
    }

    bool array(Json& out, int depth) {
        out.type = Json::Type::Array;
        ++p_;  // '['
        skip_ws();
        if (p_ < s_.size() && s_[p_] == ']') { ++p_; return true; }
        for (;;) {
            skip_ws();
            Json v;
            if (!value(v, depth + 1)) return false;
            out.array.push_back(std::move(v));
            skip_ws();
            if (p_ >= s_.size()) return bad("daftar tidak ditutup");
            if (s_[p_] == ',') { ++p_; continue; }
            if (s_[p_] == ']') { ++p_; return true; }
            return bad("',' atau ']' diharapkan");
        }
    }

    static void put_utf8(std::string& o, unsigned long cp) {
        if (cp < 0x80) o += static_cast<char>(cp);
        else if (cp < 0x800) { o += static_cast<char>(0xC0 | (cp >> 6)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { o += static_cast<char>(0xE0 | (cp >> 12)); o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
        else { o += static_cast<char>(0xF0 | (cp >> 18)); o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
    }

    bool hex4(unsigned long& v) {
        if (p_ + 4 > s_.size()) return bad("escape \\u terpotong");
        v = 0;
        for (int i = 0; i < 4; ++i) {
            const char h = s_[p_++];
            v <<= 4;
            if (h >= '0' && h <= '9') v |= static_cast<unsigned long>(h - '0');
            else if (h >= 'a' && h <= 'f') v |= static_cast<unsigned long>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') v |= static_cast<unsigned long>(h - 'A' + 10);
            else return bad("escape \\u berisi karakter bukan heksadesimal");
        }
        return true;
    }

    bool string(std::string& out) {
        ++p_;  // '"'
        out.clear();
        for (;;) {
            if (p_ >= s_.size()) return bad("teks tidak ditutup");
            const unsigned char c = static_cast<unsigned char>(s_[p_++]);
            if (c == '"') return true;
            if (c < 0x20) return bad("karakter kontrol mentah di dalam teks");
            if (c != '\\') { out += static_cast<char>(c); continue; }
            if (p_ >= s_.size()) return bad("escape terpotong");
            const char e = s_[p_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned long cp = 0;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {  // pasangan pengganti
                        unsigned long lo = 0;
                        if (s_.compare(p_, 2, "\\u") != 0) return bad("pasangan pengganti UTF-16 tidak lengkap");
                        p_ += 2;
                        if (!hex4(lo)) return false;
                        if (lo < 0xDC00 || lo > 0xDFFF) return bad("pasangan pengganti UTF-16 tidak sah");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        return bad("pengganti UTF-16 rendah tanpa pasangan");
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: return bad("escape tidak dikenal");
            }
        }
    }

    bool number(Json& out) {
        const size_t start = p_;
        bool integral = true;
        if (s_[p_] == '-') ++p_;
        if (p_ >= s_.size()) return bad("angka tidak lengkap");
        if (s_[p_] == '0') {
            ++p_;
        } else if (s_[p_] >= '1' && s_[p_] <= '9') {
            while (p_ < s_.size() && s_[p_] >= '0' && s_[p_] <= '9') ++p_;
        } else {
            return bad("angka tidak sah");
        }
        if (p_ < s_.size() && s_[p_] == '.') {
            integral = false;
            ++p_;
            if (p_ >= s_.size() || s_[p_] < '0' || s_[p_] > '9') return bad("angka: digit diharapkan setelah '.'");
            while (p_ < s_.size() && s_[p_] >= '0' && s_[p_] <= '9') ++p_;
        }
        if (p_ < s_.size() && (s_[p_] == 'e' || s_[p_] == 'E')) {
            integral = false;
            ++p_;
            if (p_ < s_.size() && (s_[p_] == '+' || s_[p_] == '-')) ++p_;
            if (p_ >= s_.size() || s_[p_] < '0' || s_[p_] > '9') return bad("angka: digit diharapkan pada eksponen");
            while (p_ < s_.size() && s_[p_] >= '0' && s_[p_] <= '9') ++p_;
        }
        const std::string tok = s_.substr(start, p_ - start);
        out.type = Json::Type::Number;
        errno = 0;
        out.number = std::strtod(tok.c_str(), nullptr);
        if (errno == ERANGE && (out.number > 1.0 || out.number < -1.0)) return bad("angka di luar jangkauan");
        out.is_integer = false;
        if (integral) {
            errno = 0;
            const long long v = std::strtoll(tok.c_str(), nullptr, 10);
            if (errno == 0) { out.is_integer = true; out.integer = v; }
        }
        return true;
    }
};

}  // namespace detail

// true bila `text` adalah satu dokumen JSON yang sah; selain itu `err` berisi alasannya.
inline bool json_parse(const std::string& text, Json& out, std::string& err) {
    detail::JsonParser parser(text);
    return parser.parse(out, err);
}

}  // namespace lc

#endif  // LANDCOVER_JSON_MINI_H
