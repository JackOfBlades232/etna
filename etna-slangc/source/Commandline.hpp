#pragma once
#ifndef ETNA_COMMANDLINE_HPP_INCLUDED
#define ETNA_COMMANDLINE_HPP_INCLUDED

#include <vector>
#include <string>
#include <span>
#include <optional>
#include <functional>

struct CommandlineArgumentName
{
  std::string primaryName{};
  std::string longName{};
};

struct CommandlineArgumentDesc
{
  CommandlineArgumentName name;
  std::string desc{};

  // Callbacks to consume the value AND report app-specific error logic
  // Should return false if one occurred
  // Define only one -- cb for flags, valueCb for args with values
  std::optional<std::function<bool()>> cb{};
  std::optional<std::function<bool(std::string_view)>> valueCb{};
};

class CommandlineParser
{
  std::vector<CommandlineArgumentDesc> args;
  std::string programName;
  std::string helpMessage;
  size_t defaultArgId;

public:
  struct CreateInfo
  {
    std::vector<CommandlineArgumentDesc> args;
    CommandlineArgumentName helpArg;
    std::string programName;
  };

  explicit CommandlineParser(CreateInfo&& ci);

  int parse(std::span<char*> argv) const;

  void reportUsageError(std::string&& error);

private:
  void showHelpMessage() const;
};

#endif
