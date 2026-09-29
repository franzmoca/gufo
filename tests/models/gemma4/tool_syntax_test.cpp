#include "src/models/gemma4/tool_syntax.hpp"

#include <string>

#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

namespace {

std::string ParseToJson(const std::string& body, std::string* name = nullptr) {
  std::string error;
  const auto call = g4::ParseToolCall(body, &error);
  Require(call.has_value(), "rejected " + body + ": " + error);
  if (name != nullptr) {
    *name = call->name;
  }
  return call->arguments.dump();
}

void CheckValues() {
  std::string name;
  Require(ParseToJson("call:get_weather{location:<|\"|>Paris, FR<|\"|>}",
                      &name) == R"({"location":"Paris, FR"})",
          "string argument");
  Require(name == "get_weather", "function name");
  Require(ParseToJson("call:mcp/github.create_issue:v2{n:1}", &name) ==
                  R"({"n":1})" &&
              name == "mcp/github.create_issue:v2",
          "namespaced function name");
  Require(ParseToJson("call:f{}") == "{}", "no arguments");
  Require(ParseToJson("call:f{n:42,x:-1.5,e:1e-05,t:true,f:false,z:null}") ==
              R"({"n":42,"x":-1.5,"e":1e-05,"t":true,"f":false,"z":null})",
          "literals");
  Require(ParseToJson("call:f{a:[1,<|\"|>two<|\"|>,[]],o:{k:{deep:1}}}") ==
              R"({"a":[1,"two",[]],"o":{"k":{"deep":1}}})",
          "nested containers");
  Require(
      ParseToJson("call:f{<|\"|>quoted key<|\"|>:1}") == R"({"quoted key":1})",
      "quoted key");
  // Strings are raw between delimiters: quotes, braces, commas, newlines.
  Require(ParseToJson("call:f{s:<|\"|>a \"b\" {c}, d\n<|\"|>}") ==
              R"({"s":"a \"b\" {c}, d\n"})",
          "raw string payload");
  Require(ParseToJson("call:f{mode:fast}") == R"({"mode":"fast"})",
          "bare word kept as string");
  Require(ParseToJson("call:f{ a : 1 , b : <|\"|>x<|\"|> }") ==
              R"({"a":1,"b":"x"})",
          "whitespace between tokens");
}

void CheckRejections() {
  for (const char* body :
       {"", "call:", "call:{}", "f{a:1}", "call:f{a:1", "call:f{a:<|\"|>x}",
        "call:f{a:1}}", "call:f{a}", "call:f{a:[1,2}", "call:f{:1}"}) {
    std::string error;
    Require(!g4::ParseToolCall(body, &error).has_value() && !error.empty(),
            std::string("accepted malformed call: ") + body);
  }
  std::string deep = "call:f{a:";
  for (int i = 0; i < 100; ++i) {
    deep += "[";
  }
  Require(!g4::ParseToolCall(deep).has_value(), "unbounded nesting accepted");
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckValues();
    CheckRejections();
  });
}
