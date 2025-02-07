// The dataset is too big to work on average laptops. So this
// program automates the task to divide the dataset into small
// pieces. The dataset can be divided into 10 equally sized
// portions.
// Compile:
// g++ -std=c++20 src/cpp/extract.cxx -o bin/extract
// Usage:
// bin/extract 1 # Generates data/Fraud1.csv with 1st decile of
// data/Fraud.csv
// bin/extract 7 # Generates data/Fraud7.csv with 7th decile of
// data/Fraud.csv
#include <exception>
#include <fstream>
#include <iostream>
#include <string>

int main(int argv, char** argc)
{
  std::string part{*(argc+1)};
  int p{};
  try
  {
    p = std::stoi(part);
  }
  catch (const std::exception& e)
  {
    std::cerr << "Abort\n";
    return 1;
  }
  if (p < 1 or p > 10)
  {
    std::cerr << "Abort\n";
    return 1;
  }
  int end{p * 636262};
  int start{end - 636262 + 1};
  std::fstream ifile{"data/Fraud.csv", std::ios::in};
  std::fstream ofile{"data/Fraud" + std::to_string(p) + ".csv", std::ios::out};
  int i{1};
  std::string line{};
  while (std::getline(ifile, line))
  {
    if (i == 1)
      ofile << line << '\n';
    if (i > start)
    {
      ofile << line << '\n';
    }
    if (i >= end)
      break;
    ++i;
  }
  ofile.close();
  ifile.close();
  return 0;
}


