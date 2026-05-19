#include "binarycore/encoding/tokenizer.hpp"

namespace binarycore {

namespace {

constexpr bool is_letter(char c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
constexpr bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }
constexpr bool is_word_char(char c) noexcept {
  return is_letter(c) || is_digit(c);
}

std::size_t match_newline(std::string_view s, std::size_t i) noexcept {
  if (i >= s.size()) return 0;
  if (s[i] == '\r') {
    if (i + 1 < s.size() && s[i + 1] == '\n') return 2;
    return 1;
  }
  if (s[i] == '\n') return 1;
  return 0;
}

void emit_symbol(std::vector<Token>& out, bits::Symbol s) {
  out.push_back(Token{encode_symbol(s), std::string{}});
}

void emit_symbol_char(std::vector<Token>& out, char c) {
  out.push_back(Token{encode_symbol_char(c), std::string(1, c)});
}

void emit_word(std::vector<Token>& out, std::string_view text) {
  out.push_back(Token{encode_word(text), std::string(text)});
}

}  // namespace

std::vector<Token> tokenize(std::string_view input) {
  std::vector<Token> out;
  out.reserve(input.size() / 4);

  const std::size_t n = input.size();
  std::size_t i = 0;

  while (i < n) {
    if (std::size_t nl = match_newline(input, i); nl > 0) {
      std::size_t count = 0;
      while (true) {
        std::size_t next = match_newline(input, i);
        if (next == 0) break;
        i += next;
        ++count;
        while (i < n && (input[i] == ' ' || input[i] == '\t')) ++i;
      }
      if (count >= 2) {
        emit_symbol(out, bits::Symbol::ParagraphBreak);
      } else {
        emit_symbol(out, bits::Symbol::Newline);
      }
      continue;
    }

    if (input[i] == ' ' || input[i] == '\t') {
      std::size_t start = i;
      while (i < n && (input[i] == ' ' || input[i] == '\t')) ++i;
      std::size_t span = i - start;

      bool has_tab = false;
      for (std::size_t k = start; k < i; ++k) {
        if (input[k] == '\t') {
          has_tab = true;
          break;
        }
      }

      if (has_tab || span >= 2) {
        emit_symbol(out, bits::Symbol::Indent);
      }
      continue;
    }

    if (is_word_char(input[i])) {
      std::size_t start = i;
      while (i < n && is_word_char(input[i])) ++i;
      emit_word(out, input.substr(start, i - start));
      continue;
    }

    char c = input[i];
    if (static_cast<unsigned char>(c) >= 0x20 &&
        static_cast<unsigned char>(c) < 0x7F) {
      emit_symbol_char(out, c);
      ++i;
      continue;
    }

    emit_symbol(out, bits::Symbol::Unknown);
    ++i;
  }

  return out;
}

}  // namespace binarycore
