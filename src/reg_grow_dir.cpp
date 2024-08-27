#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>

class Stack
{
public:
  std::vector<std::pair<int, int>> item;

  /**
   * @brief Pushes a pair of integers to the stack
   *
   * @param value The pair of integers to push
   */
  void push(std::pair<int, int> value)
  {
    item.push_back(value);
  }

  /**
   * @brief Pops a pair of integers from the stack
   *
   * @return The popped pair of integers
   *
   * Pops a pair of integers from the stack and returns it.
   */
  std::pair<int, int> pop()
  {
    std::pair<int, int> val = item.back();
    item.pop_back();
    return val;
  }

  size_t size()
  {
    return item.size();
  }

  bool isEmpty()
  {
    return item.empty();
  }

  void clear()
  {
    item.clear();
  }
};

class RegionGrow
{
public:
  cv::Mat im, passedBy, SEGS;
  int h, w;
  int currentRegion = 0;
  int iterations = 0;
  Stack stack;
  double thresh;

  /**
   * @brief Constructs a RegionGrow object
   *
   * @param img_path Path to the image to process
   * @param th Threshold value for region growing
   *
   * Constructs a RegionGrow object with the given image path and threshold.
   * It reads the image, sets the height and width of the image, allocates memory
   * for the passedBy array, initializes the currentRegion and iterations to 0,
   * initializes the SEGS Mat to zeros, and sets the threshold.
   */
  RegionGrow(const std::string &img_path, double th)
  {
    readImage(img_path);
    h = im.rows;
    w = im.cols;
    passedBy = cv::Mat::zeros(h, w, CV_64F);
    SEGS = cv::Mat::zeros(h, w, CV_8UC3);
    thresh = th;
  }

  /**
   * @brief Read an image from a file path and convert it to CV_32S
   *
   * @param img_path Path to the image
   *
   * This function reads an image from the given file path using `cv::imread`.
   * It then converts the image to the data type `CV_32S` using `cv::Mat::convertTo`.
   */
  void readImage(const std::string &img_path)
  {
    im = cv::imread(img_path, cv::IMREAD_COLOR);
    im.convertTo(im, CV_32S); // Convert to int
  }

  std::vector<std::pair<int, int>> getNeighbour(int x0, int y0)
  {
    std::vector<std::pair<int, int>> neighbours;
    for (int i = -1; i <= 1; ++i)
    {
      for (int j = -1; j <= 1; ++j)
      {
        if ((i != 0 || j != 0) && boundaries(x0 + i, y0 + j))
        {
          neighbours.push_back(std::make_pair(x0 + i, y0 + j));
        }
      }
    }
    return neighbours;
  }

  /**
   * @brief Apply region growing algorithm to image
   *
   * @param seeds The list of starting points for region growing
   * @param cv_display Whether or not to display the segmented image
   *
   * This function applies the region growing algorithm to the image.
   *
   * The algorithm starts from the given seeds and iteratively expands
   * each seed until it reaches the boundaries of the image. At each
   * iteration, the algorithm checks if the current pixel has been
   * processed before. If not, it sets the current region number,
   * pushes the pixel to the stack, marks it as processed, and performs
   * a Breadth-First Search on its neighbors. If the number of pixels
   * in the current region is less than 8 * 8, the algorithm resets
   * the current region and starts from the last seed.
   *
   * After all seeds have been processed, the function either displays
   * the segmented image or returns.
   */
  void ApplyRegionGrow(std::vector<std::pair<int, int>> &seeds, bool cv_display = true)
  {
    std::vector<std::pair<int, int>> temp;
    for (auto &i : seeds)
    {
      temp.push_back(i);
      auto neighbours = getNeighbour(i.first, i.second);
      temp.insert(temp.end(), neighbours.begin(), neighbours.end());
    }
    seeds = temp;

    for (auto &i : seeds)
    {
      int x0 = i.first;
      int y0 = i.second;

      if (passedBy.at<double>(x0, y0) == 0 && cv::norm(im.at<cv::Vec3i>(x0, y0)) > 0)
      {
        currentRegion++;
        passedBy.at<double>(x0, y0) = currentRegion;
        stack.push(std::make_pair(x0, y0));

        while (!stack.isEmpty())
        {
          auto [x, y] = stack.pop();
          BFS(x, y);
          iterations++;
        }

        if (PassedAll())
          break;

        int count = cv::countNonZero(passedBy == currentRegion);
        if (count < 8 * 8)
        {
          auto [new_x, new_y] = reset_region(x0, y0);
          x0 = new_x;
          y0 = new_y;
        }
      }
    }

    if (cv_display)
    {
      for (int i = 0; i < h; ++i)
      {
        for (int j = 0; j < w; ++j)
        {
          color_pixel(i, j);
        }
      }
      display();
    }
  }

  /**
   * @brief Reset the region growing to the previous region.
   *
   * @param x0 The x-coordinate of the starting point.
   * @param y0 The y-coordinate of the starting point.
   *
   * This function resets the region growing to the previous region. It sets the value of
   * the passedBy array to 0 where the value is equal to the currentRegion. It then decrements
   * the currentRegion by 1. It returns a pair of the previous x-coordinate and previous y-coordinate.
   */
  std::pair<int, int> reset_region(int x0, int y0)
  {
    passedBy.setTo(0, passedBy == currentRegion);
    currentRegion--;
    return std::make_pair(x0 - 1, y0 - 1);
  }

  /**
   * @brief Perform Breadth-First Search on the image starting from the given coordinates.
   *
   * @param x0 The x-coordinate of the starting point.
   * @param y0 The y-coordinate of the starting point.
   *
   * This function performs Breadth-First Search on the image starting from the given coordinates.
   * It initializes the region number to the value of the passedBy array at the starting point,
   * initializes the vector of elements with the mean of the pixel value at the starting point,
   * initializes the variance to the threshold, and gets the neighbors of the starting point.
   * It then iterates over the neighbors. If the neighbor has not been processed before and the
   * distance between the current coordinate and its neighbor is less than the variance, it sets
   * the passedBy value of the neighbor to the region number, pushes the neighbor to the stack,
   * adds the mean of the pixel value at the neighbor to the vector of elements, and updates the
   * variance to the mean of the vector of elements. If the number of pixels in the current region
   * is less than 8 * 8, the function resets the region to the previous region.
   */
  void BFS(int x0, int y0)
  {
    double regionNum = passedBy.at<double>(x0, y0);
    std::vector<double> elems = {cv::mean(im.at<cv::Vec3i>(x0, y0))[0]};

    double var = thresh;
    auto neighbours = getNeighbour(x0, y0);

    for (auto &[x, y] : neighbours)
    {
      if (passedBy.at<double>(x, y) == 0 && distance(x, y, x0, y0) < var)
      {
        if (PassedAll())
          break;

        passedBy.at<double>(x, y) = regionNum;
        stack.push(std::make_pair(x, y));
        elems.push_back(cv::mean(im.at<cv::Vec3i>(x, y))[0]);
        var = cv::mean(elems)[0];
      }
      var = std::max(var, thresh);
    }
  }

  /**
   * @brief Set the color of a pixel in the segmented image
   *
   * @param i The x-coordinate of the pixel
   * @param j The y-coordinate of the pixel
   *
   * This function sets the color of a pixel in the segmented image. If the pixel
   * is not part of a region, it sets the color to white. Otherwise, it scales
   * the region number to values between 35 and 90 for the red channel, and
   * values between 30 and 270 for the green and blue channels.
   */
  void color_pixel(int i, int j)
  {
    double val = passedBy.at<double>(i, j);
    SEGS.at<cv::Vec3b>(i, j) = (val == 0) ? cv::Vec3b(255, 255, 255) : cv::Vec3b(val * 35, val * 90, val * 30);
  }

  /**
   * @brief Display the segmented image
   *
   * This function displays the segmented image using `cv::imshow`.
   * It waits for a key press using `cv::waitKey` and then destroys
   * all windows using `cv::destroyAllWindows`.
   */
  void display()
  {
    cv::imshow("Segmented Image", SEGS);
    cv::waitKey(0);
    cv::destroyAllWindows();
  }

  /**
   * @brief Check if the region growing algorithm has passed all pixels
   *
   * @param max_iteration The maximum number of iterations to perform
   *
   * @return True if either the maximum number of iterations has been reached,
   * or all pixels have been assigned to a region. False otherwise.
   *
   * This function checks if the region growing algorithm has passed all pixels
   * by comparing the number of iterations to the maximum number of iterations
   * and the number of pixels that have been assigned to a region to the total
   * number of pixels in the image.
   */
  bool PassedAll(int max_iteration = 200000)
  {
    return iterations > max_iteration || cv::countNonZero(passedBy > 0) == h * w;
  }

  /**
   * @brief Check if the given coordinates are within the boundaries of the
   * image.
   *
   * @param x The x-coordinate to check
   * @param y The y-coordinate to check
   *
   * @return True if the coordinates are within the image boundaries, false
   * otherwise.
   *
   * This function checks if the given coordinates are within the boundaries of
   * the image by comparing the coordinates to the height and width of the image.
   */
  bool boundaries(int x, int y)
  {
    return x >= 0 && x < h && y >= 0 && y < w;
  }

  double distance(int x, int y, int x0, int y0)
  {
    return cv::norm(im.at<cv::Vec3i>(x0, y0) - im.at<cv::Vec3i>(x, y));
  }
};

std::vector<std::pair<int, int>> seeds;

/**
 * @brief Callback function for mouse events in the image window.
 *
 * @param event The type of mouse event that occurred
 * @param x The x-coordinate of the mouse event
 * @param y The y-coordinate of the mouse event
 * @param, void * unused
 *
 * This function adds the coordinates of a mouse click to the seeds vector if the
 * left mouse button is pressed. It closes all windows if the right mouse button is
 * pressed.
 */
void get_seeds(int event, int x, int y, int, void *)
{
  if (event == cv::EVENT_LBUTTONDOWN)
  {
    seeds.push_back(std::make_pair(y, x));
  }
  else if (event == cv::EVENT_RBUTTONDOWN)
  {
    cv::destroyAllWindows();
  }
}

/**
 * @brief Main function for the program.
 *
 * @param argc The number of command line arguments.
 * @param argv An array of strings containing the command line arguments.
 *
 * This function reads an image from the given file path using `cv::imread`.
 * It then converts the image to the data type `CV_32S` using `cv::Mat::convertTo`.
 * It initializes a RegionGrow object with the given image path and threshold.
 * It creates a window named "image" and sets a callback function for mouse events.
 * It shows the image in the window. It waits for a key press.
 * It applies the region growing algorithm to the image with the given seeds.
 * It displays the segmented image.
 *
 * @return 0 upon successful completion.
 *
 * @throws std::invalid_argument if the number of command line arguments is not 3.
 */
int main(int argc, char **argv)
{
  if (argc != 3)
  {
    std::cerr << "Usage: " << argv[0] << " <image_path> <threshold>" << std::endl;
    return -1;
  }

  std::string img_path = argv[1];
  double thresh = std::stod(argv[2]);

  RegionGrow exemple(img_path, thresh);

  cv::namedWindow("image");
  cv::setMouseCallback("image", get_seeds);
  cv::imshow("image", cv::imread(img_path, cv::IMREAD_COLOR));
  cv::waitKey(0);

  exemple.ApplyRegionGrow(seeds);

  return 0;
}
