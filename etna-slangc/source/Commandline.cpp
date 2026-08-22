#include "Commandline.hpp"

#include <spdlog/spdlog.h>
#include <fmt/format.h>

#include <unordered_set>

CommandlineParser::CommandlineParser(CreateInfo&& ci)
  : args{std::move(ci.args)}
  , programName{std::move(ci.programName)}
{
  args.push_back(
    CommandlineArgumentDesc{
      .name = std::move(ci.helpArg),
      .kind = CommandlineArgumentKind::FLAG,
      .desc = "display help message",
      .cb = [this] {
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

    if (arg.kind == CommandlineArgumentKind::FLAG)
      assert(arg.cb.has_value());
    else if (arg.kind == CommandlineArgumentKind::STRING)
      assert(arg.valueCb.has_value());
    helpMessage += "  ";
    if (arg.name.fullName.empty())
    {
      assert(arg.name.shortName.empty());
      assert(arg.kind == CommandlineArgumentKind::STRING);
      defaultArgId = i;
      helpMessage += "<any arg>";
    }
    else
    {
      helpMessage += arg.name.fullName;
      if (!arg.name.shortName.empty())
        helpMessage += fmt::format(" ({})", arg.name.shortName);
    }
    if (!arg.desc.empty())
      helpMessage += fmt::format(" -- {}", arg.desc);
    helpMessage += "\n";

    ++totalNameCount;
    uniqueNames.insert(arg.name.fullName);
    if (!arg.name.shortName.empty())
    {
      ++totalNameCount;
      uniqueNames.insert(arg.name.shortName);
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
      if (arg.name.fullName.empty())
        continue;

      bool match = strcmp(arg.name.fullName.c_str(), argv[i]) == 0;
      if (!match && !arg.name.shortName.empty())
        match = strcmp(arg.name.shortName.c_str(), argv[i]) == 0;
      if (!match)
        continue;

      switch (arg.kind)
      {
      case CommandlineArgumentKind::FLAG: {
        if (!arg.cb.value()())
        {
          spdlog::error("{}", helpMessage);
          return 1;
        }
        break;
      }
      case CommandlineArgumentKind::STRING: {
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
        break;
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
