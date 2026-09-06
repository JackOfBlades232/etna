#include "Commandline.hpp"

#include <spdlog/spdlog.h>
#include <fmt/format.h>

#include <unordered_set>

CommandlineParser::CommandlineParser(CreateInfo&& ci)
  : args{std::move(ci.args)}
  , programName{std::move(ci.programName)}
{
  args.push_back(CommandlineArgumentDesc{
    .name = std::move(ci.helpArg), .desc = "display help message", .cb = [this] {
      showHelpMessage();
      exit(0);
      return true;
    }});

  std::unordered_set<std::string> uniqueNames{};
  size_t totalNameCount = 0;

  defaultArgId = args.size();

  size_t i = 0;
  helpMessage = "";
  helpMessage += "Usage:\n";
  for (const auto& arg : args)
  {
    bool isFlag = arg.cb.has_value();
    bool isString = arg.valueCb.has_value();
    assert(isFlag || isString);
    assert(!isFlag || !isString);
    helpMessage += "  ";
    if (arg.name.primaryName.empty())
    {
      assert(arg.name.longName.empty());
      assert(isString);
      defaultArgId = i;
      helpMessage += "<any arg>";
    }
    else
    {
      helpMessage += arg.name.primaryName;
      if (!arg.name.longName.empty())
        helpMessage += fmt::format(" ({})", arg.name.longName);
    }
    if (!arg.desc.empty())
      helpMessage += fmt::format(" -- {}", arg.desc);
    helpMessage += "\n";

    ++totalNameCount;
    uniqueNames.insert(arg.name.primaryName);
    if (!arg.name.longName.empty())
    {
      ++totalNameCount;
      uniqueNames.insert(arg.name.longName);
    }

    ++i;
  }

  assert(uniqueNames.size() == totalNameCount);
}

int CommandlineParser::parse(std::span<char*> argv) const
{
  for (size_t i = 0; i < argv.size(); ++i)
  {
    bool handled = false;
    for (const auto& arg : args)
    {
      if (arg.name.primaryName.empty())
        continue;

      bool match = strcmp(arg.name.primaryName.c_str(), argv[i]) == 0;
      if (!match && !arg.name.longName.empty())
        match = strcmp(arg.name.longName.c_str(), argv[i]) == 0;
      if (!match)
        continue;

      if (arg.cb.has_value())
      {
        if (!arg.cb.value()())
        {
          spdlog::error("{}", helpMessage);
          return 1;
        }
      }
      else if (arg.valueCb.has_value())
      {
        if (i >= argv.size() - 1)
        {
          spdlog::error("Invalid usage: {} arg requires a string value", argv[i]);
          spdlog::error("{}", helpMessage);
          return 1;
        }
        ++i;
        if (!arg.valueCb.value()(argv[i]))
        {
          spdlog::error("{}", helpMessage);
          return 1;
        }
      }

      handled = true;
      break;
    }

    if (!handled)
    {
      if (defaultArgId < args.size())
      {
        if (!args[defaultArgId].valueCb.value()(argv[i]))
        {
          spdlog::error("{}", helpMessage);
          return 1;
        }
      }
      else
      {
        spdlog::error("Invalid usage: unknown argument {}", argv[i]);
        spdlog::error("{}", helpMessage);
        return 1;
      }
    }
  }
  return 0;
}


void CommandlineParser::reportUsageError(std::string&& error)
{
  spdlog::error("Invalid usage: {}", error);
  spdlog::error("{}", helpMessage);
}

void CommandlineParser::showHelpMessage() const
{
  spdlog::info("--- {} ---", programName);
  spdlog::info("{}", helpMessage);
}
