#pragma once

class SUTILCLASSAPI VCAOptions
{
public:
  std::string    url;
  std::string    user;
  std::string    password;
  int            num_nodes;
  int            config_index;

  VCAOptions() :
    num_nodes(1),
    config_index(0)
  {}
};

