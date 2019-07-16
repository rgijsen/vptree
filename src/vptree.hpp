// Based on http://stevehanov.ca/blog/index.php?id=130 by Steve Hanov

#pragma once

#include <algorithm>
#include <vector>
#include <queue>
#include <limits>
#include <random>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace vpt
{
  // get the dimension of the position vector.
  // this function allows the container to be of aggregated type
  template<typename Vector>
  using GetDimension = std::function<size_t(const Vector&)>;
  // in case the container is an aggregated type, this function extracts the position vector from the container
  template<typename Vector, typename AggregateVector = Vector>
  using GetAggregateFromContainer = std::function<AggregateVector(const Vector&)>;
  // metric used when calculating the distance between points
  template<typename Vector>
  using Metric = std::function<double(const Vector& v1, const Vector& v2)>;
  using DistancesIndices = std::pair<std::vector<double>, std::vector<int>>;
  using BatchDistancesIndices = std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>>;

  // full device euclidean metric
  template<typename Container, typename AggregateVector = Container>
  struct EuclideanMetric 
  {
    GetAggregateFromContainer<Container, AggregateVector> getAggregate;

    EuclideanMetric(GetAggregateFromContainer<Container, AggregateVector> getAggr = [](auto v) { return v; })
      : getAggregate(getAggr)
    {}

    double operator() (const Container& v1, const Container& v2) const 
    {
      auto l1 = getAggregate(v1);
      auto l2 = getAggregate(v2);

      decltype(l1) diff(l1.size());
      for(int i(0); i<l1.size(); ++i)
      {
        diff[i] = l1[i] - l2[i];
      }

      // euclidean norm
      return std::sqrt(std::inner_product(diff.data(), diff.data() + diff.size(), diff.data(), 0.0));
    }
  };

  // periodic boundary condition euclidean metric
  template<typename Container, typename Dimensions, typename Periodicity, typename AggregateVector = Container>
  struct EuclideanMetricPbc 
  {
    Dimensions dimensions;
    Periodicity periodic;
    GetAggregateFromContainer<Container, AggregateVector> getAggregate;

    EuclideanMetricPbc(Dimensions dims, 
                        Periodicity p, 
                        GetAggregateFromContainer<Container, AggregateVector> getAggr = [](auto v) { return v; })
      : dimensions(dims), periodic(p), getAggregate(getAggr)
    {}

    double operator() (const Container& v1, const Container& v2) const 
    {
      auto l1 = getAggregate(v1);
      auto l2 = getAggregate(v2);

      decltype(l1) diff(l1.size());
      for(int i(0); i<l1.size(); ++i)
      {
        if(periodic[i]) // not all dimensions have to be periodic, so check
        {
          auto d = l1[i] - l2[i];
          auto dim = dimensions[i];
          auto hdim = (double)dim / 2;
          if(d < -hdim) d += dim; else if(d > hdim) d -= dim;
          diff[i] = d;
        }
        else
          diff[i] = l1[i] - l2[i];
      }

      // euclidean norm
      return std::sqrt(std::inner_product(diff.data(), diff.data() + diff.size(), diff.data(), 0.0));
    }
  };

  class DimensionMismatch: public std::runtime_error 
  {
    public:
      DimensionMismatch(int expected, int got)
      : std::runtime_error("Item dimension doesn't match: expected " + std::to_string(expected) + ", got " + std::to_string(got))
      {}
  };

  template<typename Vector>
  class Searcher;

  template<typename Vector = std::vector<double>>
  class VpTree 
  {
    public:
      template<typename InputIterator>
      explicit VpTree(InputIterator start, InputIterator end, 
                      Metric<Vector> metric = EuclideanMetric<Vector>(), 
                      GetDimension<Vector> getDimension = [](auto v) { return v.size(); });

      template<typename Container>
      explicit VpTree(const Container& container, 
                      Metric<Vector> metric = EuclideanMetric<Vector>(),
                      GetDimension<Vector> getDimension = [](auto v) { return v.size(); });
      // explicit VpTree(std::initializer_list<Vector> list,
      //                 Metric<std::initializer_list<Vector>> metric = EuclideanMetric<std::initializer_list<Vector>>());

      DistancesIndices getNearestNeighbors(const Vector& target, int neighborsCount) const;
      template<typename VectorLike>
      DistancesIndices getNearestNeighbors(const VectorLike& target, int neighborsCount) const;
      DistancesIndices getNearestNeighbors(std::initializer_list<double> target, int neighborsCount) const;

      template<typename Container>
      BatchDistancesIndices getNearestNeighborsBatch(const Container& targets, int neighborsCount) const;
      BatchDistancesIndices getNearestNeighborsBatch(std::initializer_list<Vector> targets, int neighborsCount) const;

      DistancesIndices getNeighborhoodPoints(const Vector& target, double radius) const;

    private:
      const Metric<Vector> getDistance;
      const GetDimension<Vector> getDimension;

      struct Node
      {
        static const int Leaf = -1;

        Node(int item, double threshold = 0., int left = Leaf, int right = Leaf)
        : item(item), threshold(threshold), left(left), right(right)
        { }

        int item;
        double threshold;
        int left;
        int right;
      };

    private:
      typedef std::pair<Vector, int> Item;

      std::vector<Item> items_;
      std::vector<Node> nodes_;

      std::mt19937 rng_;

      int dimension_;

    private:
      template<typename InputIterator>
      std::vector<Item> makeItems(InputIterator start, InputIterator end);

      int makeTree(int lower, int upper);
      void selectRoot(int lower, int upper);
      void partitionByDistance(int lower, int pos, int upper);
      int makeNode(int item);
      Node root() const { return nodes_[0]; }

      friend class Searcher<Vector>;
  };

  template<typename Vector>
  class Searcher 
  {
    private:
      typedef typename VpTree<Vector>::Node Node;

    public:
      explicit Searcher(const VpTree<Vector>* tree, const Vector& target, int neighborsCount);
      explicit Searcher(const VpTree<Vector>* tree, const Vector& target, double radius);

      DistancesIndices search();

      struct HeapItem 
      {
          bool operator < (const HeapItem& other) const 
          {
              return dist < other.dist;
          }

          int item;
          double dist;
      };

    private:
      void searchInNode(const Node& node);

      const VpTree<Vector>* tree_;
      Vector target_;
      int neighborsCount_;
      double radius_;
      double tau_;
      std::priority_queue<HeapItem> heap_;
  };

  template<typename Vector>
  template<typename InputIterator>
  VpTree<Vector>::VpTree(InputIterator start, InputIterator end, Metric<Vector> metric, GetDimension<Vector> getDim)
    : getDistance(metric), items_(makeItems(start, end)), nodes_(), rng_(), getDimension(getDim)
  {
      std::random_device rd;
      rng_.seed(rd());
      nodes_.reserve(items_.size());
      makeTree(0, items_.size());
  }

  template<typename Vector>
  template<typename Container>
  VpTree<Vector>::VpTree(const Container& container, Metric<Vector> metric, GetDimension<Vector> getDim)
    : VpTree(container.begin(), container.end(), metric, getDim)
  { }

  // template<typename Vector>
  // VpTree<Vector>::VpTree(std::initializer_list<Vector> list, Metric<std::initializer_list<Vector>> metric)
  //   : VpTree(list.begin(), list.end(), metric)
  // { }

  template<typename Vector>
  int VpTree<Vector>::makeTree(int lower, int upper) 
  {
    if (lower >= upper) 
    {
      return Node::Leaf;
    }
    else if (lower + 1 == upper)
    {
      return makeNode(lower);
    }
    else
    {
      selectRoot(lower, upper);
      int median = (upper + lower) / 2;
      partitionByDistance(lower, median, upper);
      auto node = makeNode(lower);
      nodes_[node].threshold = getDistance(items_[lower].first, items_[median].first);
      nodes_[node].left = makeTree(lower + 1, median);
      nodes_[node].right = makeTree(median, upper);
      return node;
    }
  }

  template<typename Vector>
  void VpTree<Vector>::selectRoot(int lower, int upper) {
      std::uniform_int_distribution<int> uni(lower, upper - 1);
      int root = uni(rng_);
      std::swap(items_[lower], items_[root]);
  }

  template<typename Vector>
  void VpTree<Vector>::partitionByDistance(int lower, int pos, int upper) {
      std::nth_element(
          items_.begin() + lower + 1,
          items_.begin() + pos,
          items_.begin() + upper,
          [lower, this] (const Item& i1, const Item& i2) {
              return getDistance(items_[lower].first, i1.first) < getDistance(items_[lower].first, i2.first);
          });
  }

  template<typename Vector>
  int VpTree<Vector>::makeNode(int item) {
      nodes_.push_back(Node(item));
      return nodes_.size() - 1;
  }

  template<typename Vector>
  template<typename InputIterator>
  std::vector<std::pair<Vector, int>> VpTree<Vector>::makeItems(InputIterator begin, InputIterator end) 
  {
    if (begin != end) 
    {
      dimension_ = getDimension(*begin);
    }
    else 
    {
      dimension_ = -1;
    }

    std::vector<Item> res;
    for (int i = 0; begin != end; ++begin, ++i) 
    {
      auto vec = *begin;
      res.push_back(std::make_pair(vec, i));

      auto lastDimension = getDimension(res.back().first);
      if (lastDimension != dimension_) 
      {
        throw DimensionMismatch(dimension_, lastDimension);
      }
    }
    return res;
  }

  template<typename Vector>
  template<typename VectorLike>
  DistancesIndices VpTree<Vector>::getNearestNeighbors(const VectorLike& target, int neighborsCount) const {
      return getNearestNeighbors(Vector(target.begin(), target.end()), neighborsCount);
  }

  template<typename Vector>
  DistancesIndices VpTree<Vector>::getNearestNeighbors(std::initializer_list<double> target, int neighborsCount) const {
      return getNearestNeighbors(Vector(target.begin(), target.end()), neighborsCount);
  }

  template<typename Vector>
  DistancesIndices VpTree<Vector>::getNearestNeighbors(const Vector& target, int neighborsCount) const {
      auto targetDimension = getDimension(target);
      if (targetDimension != dimension_) {
          throw DimensionMismatch(dimension_, targetDimension);
      }
      Searcher searcher(this, target, neighborsCount);
      return searcher.search();
  }

  template<typename Vector>
  DistancesIndices VpTree<Vector>::getNeighborhoodPoints(const Vector& target, double radius) const 
  {
    auto targetDimension = getDimension(target);
    if (targetDimension != dimension_)
    {
      throw DimensionMismatch(dimension_, targetDimension);
    }
    Searcher searcher(this, target, radius);
    return searcher.search();
  }

  template<typename Vector>
  template<typename Container>
  BatchDistancesIndices VpTree<Vector>::getNearestNeighborsBatch(const Container& targets, int neighborsCount) const 
  {
    std::vector<std::vector<double>> batchDistances(targets.size());
    std::vector<std::vector<int>> batchIndices(targets.size());
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < targets.size(); ++i) 
    {
      std::tie(batchDistances[i], batchIndices[i]) = getNearestNeighbors(targets[i], neighborsCount);
    }
    return BatchDistancesIndices(batchDistances, batchIndices);
  }

  template<typename Vector>
  BatchDistancesIndices VpTree<Vector>::getNearestNeighborsBatch(std::initializer_list<Vector> targets, int neighborsCount) const 
  {
    return getNearestNeighborsBatch(std::vector<Vector>(targets.begin(), targets.end()), neighborsCount);
  }

  template<typename Vector>
  Searcher<Vector>::Searcher(const VpTree<Vector>* tree, const Vector& target, int neighborsCount)
    : tree_(tree), target_(target), neighborsCount_(neighborsCount), tau_(std::numeric_limits<double>::max()), heap_(), radius_(0.0)
  { }

  template<typename Vector>
  Searcher<Vector>::Searcher(const VpTree<Vector>* tree, const Vector& target, double radius)
    : tree_(tree), target_(target), radius_(radius), tau_(std::numeric_limits<double>::max()), heap_(), neighborsCount_(0)
  { 
    tau_ = radius;    
  }

  template<typename Vector>
  DistancesIndices Searcher<Vector>::search()
  {
    searchInNode(tree_->root());

    DistancesIndices results;
    while(!heap_.empty()) 
    {
        results.first.push_back(heap_.top().dist);
        results.second.push_back(tree_->items_[heap_.top().item].second);
        heap_.pop();
    }
    std::reverse(results.first.begin(), results.first.end());
    std::reverse(results.second.begin(), results.second.end());
    return results;
  }

  template<typename Vector>
  void Searcher<Vector>::searchInNode(const Node& node)
  {
    double dist = tree_->getDistance(tree_->items_[node.item].first, target_);

    if (dist < tau_) 
    {
      if(neighborsCount_)
      {
        if (heap_.size() == neighborsCount_)
          heap_.pop();

        heap_.push(HeapItem{node.item, dist});

        if (heap_.size() == neighborsCount_)
          tau_ = heap_.top().dist;
      }
      else
      {
        if(dist <= radius_)
          heap_.push(HeapItem{node.item, dist});
      }
    }

    if (dist < node.threshold) 
    {
      if (node.left != Node::Leaf && dist - tau_ <= node.threshold)
        searchInNode(tree_->nodes_[node.left]);

      if (node.right != Node::Leaf && dist + tau_ >= node.threshold)
        searchInNode(tree_->nodes_[node.right]);
    }
    else
    {
      if (node.right != Node::Leaf && dist + tau_ >= node.threshold)
        searchInNode(tree_->nodes_[node.right]);

      if (node.left != Node::Leaf && dist - tau_ <= node.threshold)
        searchInNode(tree_->nodes_[node.left]);
    }
  }
}