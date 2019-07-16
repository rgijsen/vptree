
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include "assert.h"
#include "vptree.hpp"

namespace
{
  auto split(std::string s)
  {
    std::istringstream iss(s);
    std::vector<std::string> tokens(std::istream_iterator<std::string>{iss},
                                    std::istream_iterator<std::string>());

    return tokens;
  }

  auto create_points(std::vector<int> dimensions)
  {
    std::vector<std::vector<int>> points;

    for(int x(0); x < dimensions[0]; ++x)
      for(int y(0); y < dimensions[0]; ++y)
        for(int z(0); z < dimensions[0]; ++z)
          points.push_back({x, y, z});

    return points;
  }

  auto parse_benckmark_file(std::string file)
  {
    std::map<int, std::vector<int>> benchmark_treeIndices;

    std::string line;
    auto stream = std::ifstream(file);
    std::getline(stream, line); // skip header line
    while(std::getline(stream, line))
    {
      auto tokens = split(line);
      auto ref_idx = std::stoi(tokens[0]);
      auto neighbor_idx = std::stoi(tokens[1]);
      benchmark_treeIndices[ref_idx].push_back(neighbor_idx);
    }

    return benchmark_treeIndices;
  }
}

void BasicEuclideanMetric_test()
{
  auto points = std::vector<std::vector<double>>
  {
      {0, 0, 1},
      {1, 1, 1},
      {2, 0, 0},
      {-1, -1, 0},
      {10, 0, 5}
  };

  auto metric = vpt::EuclideanMetric<std::vector<double>>();
  auto tree = vpt::VpTree<std::vector<double>>(points, metric);
  auto [distances, treeIndices] = tree.getNearestNeighbors({ 0, 0, 0 }, 3);

  std::vector<int> benchmark_treeIndices = { 0, 1, 3 };

  std::cout << std::boolalpha
            << "BasicEuclideanMetric_test: "
            << (std::is_permutation(treeIndices.begin(), treeIndices.end(), benchmark_treeIndices.begin(), benchmark_treeIndices.end()))
            << std::endl;
}

void BasicEuclideanMetric_AggregatedType_test()
{
  using point_t = std::vector<double>;

  struct pointEx_t
  {
    int dummy;
    point_t point;

    pointEx_t(std::initializer_list<double> p)
      : point(p)
    {}
  };

  auto points = std::vector<pointEx_t>
  {
      {0, 0, 1},
      {1, 1, 1},
      {2, 0, 0},
      {-1, -1, 0},
      {10, 0, 5}
  };

  auto metric = vpt::EuclideanMetric<pointEx_t, point_t>([](auto& v) { return v.point; });
  auto tree = vpt::VpTree<pointEx_t>(points, metric, [](auto& v) { return v.point.size(); });
  auto [distances, treeIndices] = tree.getNearestNeighbors((pointEx_t){ 0, 0, 0 }, 3);

  std::vector<int> benchmark_treeIndices = { 0, 1, 3 };

  std::cout << std::boolalpha
            << "BasicEuclideanMetric_AggregatedType_test: "
            << (std::is_permutation(treeIndices.begin(), treeIndices.end(), benchmark_treeIndices.begin(), benchmark_treeIndices.end()))
            << std::endl;
}

void cubic2_pbc_test()
{
  using point_t = std::vector<int>;

  auto cutoff_radius = 1.2;
  std::vector<int> dimensions = { 2, 2, 2 };
  std::vector<bool> periodicity = { true, true, true };
  
  auto points = std::vector<point_t>
  {
      {0, 0, 0},
      {0, 0, 1},
      {0, 1, 0},
      {0, 1, 1},
      {1, 0, 0},
      {1, 1, 0},
      {1, 1, 1},
      {1, 0, 1}
  };
  
  auto metric = vpt::EuclideanMetricPbc<point_t, std::vector<int>, std::vector<bool>>(dimensions, periodicity);
  auto tree = vpt::VpTree<std::vector<int>>(points, metric);
  auto [treeDistances, treeIndices] = tree.getNeighborhoodPoints(points[0], cutoff_radius);

  std::vector<int> benchmark_treeIndices = { 0, 1, 2, 4 };

  std::cout << std::boolalpha
            << "cubic2_pbc_test: "
            << (std::is_permutation(treeIndices.begin(), treeIndices.end(), benchmark_treeIndices.begin(), benchmark_treeIndices.end()))
            << std::endl;
}

void cubic10_pbc_test()
{
  using point_t = std::vector<int>;

  auto cutoff_radius = 1.2;
  std::vector<int> dimensions = { 10, 10, 10 };
  std::vector<bool> periodicity = { true, true, true };

  auto points = create_points(dimensions);

  auto metric = vpt::EuclideanMetricPbc<point_t, std::vector<int>, std::vector<bool>>(dimensions, periodicity);
  auto tree = vpt::VpTree<point_t>(points, metric);
  
  std::string benchmark_file = "cubic10_pbc_benchmark.txt";
  auto benchmark_results = parse_benckmark_file(benchmark_file);

  for(size_t idx(0); idx < points.size(); ++idx)
  {
    auto [treeDistances, treeIndices] = tree.getNeighborhoodPoints(points[idx], cutoff_radius);
    // remove site itself from results and all previous combinations
    treeIndices.erase(std::remove_if(treeIndices.begin(), treeIndices.end(), [idx](auto a) { return a <= idx; }), treeIndices.end());
    if(idx == points.size() - 1)
    {
      assert(treeIndices.size() == 0 && "last site should not contain any new idexes");
    }
    else
    {
      assert(benchmark_results.find(idx) != benchmark_results.end() && "point not present in benchmark data");
      assert(std::is_permutation(treeIndices.begin(), treeIndices.end(), 
                                  benchmark_results[idx].begin(), benchmark_results[idx].end()) && "points not matching benchmark data");
    }
  }

  std::cout << "cubic10_pbc_test passed" << std::endl;
}
