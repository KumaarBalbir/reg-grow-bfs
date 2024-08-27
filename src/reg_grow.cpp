#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <time.h>

using namespace cv;

// stack of pixel coordinates during the region-growing process
typedef struct Stack
{
  int *x;
  int *y;
  int size;
  int capacity;
} Stack;

/**
 * @brief Initialize a stack
 *
 * @param stack pointer to a stack
 * @param capacity capacity of the stack
 *
 * Initializes a stack with given capacity.
 */
void initStack(Stack *stack, int capacity)
{
  stack->x = (int *)malloc(capacity * sizeof(int));
  stack->y = (int *)malloc(capacity * sizeof(int));
  stack->size = 0;
  stack->capacity = capacity;
}

/**
 * @brief Pushes values to stack
 *
 * @param stack pointer to a stack
 * @param x_coord x coordinate to push
 * @param y_coord y coordinate to push
 *
 * Pushes an (x_coord, y_coord) pair to the stack. If the stack is full, it
 * expands the capacity.
 */
void push(Stack *stack, int x_coord, int y_coord)
{
  if (stack->size == stack->capacity)
  {
    stack->capacity *= 2; // double the capacity if the stack is full
    stack->x = (int *)realloc(stack->x, stack->capacity * sizeof(int));
    stack->y = (int *)realloc(stack->y, stack->capacity * sizeof(int));
  }
  stack->x[stack->size] = x_coord;
  stack->y[stack->size] = y_coord;
  stack->size++;
}

/**
 * @brief Pops a value from the stack
 *
 * @param stack Pointer to the stack
 * @param x_coord Pointer to store the popped x value
 * @param y_coord Pointer to store the popped y value
 *
 * Pops a value from the stack and stores it in x_coord and y_coord. If the stack is empty,
 * does nothing.
 */
void pop(Stack *stack, int *x_coord, int *y_coord)
{
  if (stack->size > 0)
  {
    stack->size--;
    *x_coord = stack->x[stack->size];
    *y_coord = stack->y[stack->size];
  }
  else
  {
    fprintf(stderr, "Stack is empty\n");
  }
}

/**
 * @brief Check if the stack is empty
 *
 * @param stack Pointer to the stack
 * @return 1 if the stack is empty, 0 otherwise
 *
 * Returns 1 if the stack is empty, 0 otherwise.
 */
int isEmpty(Stack *stack)
{
  return stack->size == 0;
}

/**
 * @brief Free the memory allocated for stack
 *
 * @param stack Pointer to the stack
 *
 * Frees the memory allocated for the stack.
 */
void freeStack(Stack *stack)
{
  free(stack->x);
  free(stack->y);
}

typedef struct RegionGrow
{
  Mat im;            // original image
  int h, w;          // height and width
  double *passedBy;  // which pixels have already been processed
  int currentRegion; // current region number
  int iterations;
  Mat SEGS;     // stores the segmented image
  Stack stack;  // stack of pixel coordinates
  float thresh; // threshold
} RegionGrow;

/**
 * @brief Initialize a RegionGrow object
 *
 * @param rg Pointer to a RegionGrow object to initialize
 * @param img_path Path to the image to process
 * @param th Threshold value for region growing
 *
 * Initializes a RegionGrow object with the given image path and threshold.
 * It reads the image, sets the height and width of the image, allocates memory
 * for the passedBy array, initializes the currentRegion and iterations to 0,
 * initializes the SEGS Mat to zeros, initializes the stack with a capacity of
 * 1000, and sets the threshold.
 */
void initRegionGrow(RegionGrow *rg, const char *img_path, float th)
{
  rg->im = imread(img_path, IMREAD_COLOR);
  rg->h = rg->im.rows;
  rg->w = rg->im.cols;
  rg->passedBy = (double *)calloc(rg->h * rg->w, sizeof(double)); // allocates memory for the passedBy array
  rg->currentRegion = 0;
  rg->iterations = 0;
  rg->SEGS = Mat::zeros(rg->h, rg->w, CV_8UC3); // initializes the SEGS Mat to zeros
  initStack(&rg->stack, 1000);                  // TODO: which is the right capacity?
  rg->thresh = th;
}

/**
 * @brief Free the memory allocated for RegionGrow object
 *
 * @param rg Pointer to the RegionGrow object
 *
 * Frees the memory allocated for the passedBy array and the stack.
 */
void freeRegionGrow(RegionGrow *rg)
{
  free(rg->passedBy);
  freeStack(&rg->stack);
}

/**
 * @brief Check if the given coordinates are within the boundaries of the
 * image.
 *
 * @param rg Pointer to a RegionGrow object
 * @param x The x-coordinate to check
 * @param y The y-coordinate to check
 *
 * @return True if the coordinates are within the image boundaries, false
 * otherwise.
 */
int boundaries(RegionGrow *rg, int x, int y)
{
  return x >= 0 && x < rg->h && y >= 0 && y < rg->w;
}

/**
 * @brief Calculate the Euclidean distance between two 3-channel color
 * pixels.
 *
 * @param a The first color pixel
 * @param b The second color pixel
 *
 * @return The Euclidean distance between the two color pixels
 *
 * This function calculates the Euclidean distance between two 3-channel color
 * pixels. It compares the RGB values of each pixel and calculates the
 * Euclidean distance between them. The function uses the formula
 * sqrt((a_R - b_R)^2 + (a_G - b_G)^2 + (a_B - b_B)^2), where a_R, a_G, and
 * a_B are the RGB values of pixel a, and b_R, b_G, and b_B are the RGB values
 * of pixel b.
 */
double distance(Vec3b a, Vec3b b)
{
  double dist = 0;
  for (int i = 0; i < 3; i++)
  {
    dist += pow(a[i] - b[i], 2);
  }
  return sqrt(dist);
}

/**
 * @brief Perform a breadth-first search on a RegionGrow object starting from
 * the given coordinates.
 *
 * @param rg Pointer to a RegionGrow object
 * @param x0 The x-coordinate of the starting point
 * @param y0 The y-coordinate of the starting point
 *
 * This function performs a breadth-first search on a RegionGrow object starting
 * from the given coordinates. It initializes the variable `var` with the
 * threshold of the RegionGrow object, and sets the `regionNum` to the value of
 * the passedBy array at the starting point. It then enters a loop until the
 * stack is empty. In each iteration, it pops a coordinate from the stack and
 * increments the `iterations` counter. It then checks the neighbors of the
 * current coordinate and adds them to the stack if they are within the image
 * boundaries and have not been passed by before. If the distance between the
 * current coordinate and its neighbor is less than `var`, it sets the passedBy
 * value of the neighbor to `regionNum` and adds it to the stack.
 */
void BFS(RegionGrow *rg, int x0, int y0)
{
  int x, y;
  double var = rg->thresh;
  double regionNum = rg->passedBy[x0 * rg->w + y0];

  while (!isEmpty(&rg->stack))
  {
    pop(&rg->stack, &x, &y);
    rg->iterations++;

    for (int i = -1; i <= 1; i++)
    {
      for (int j = -1; j <= 1; j++)
      {
        if (i == 0 && j == 0) // centre pixel
          continue;
        int nx = x + i, ny = y + j;
        if (boundaries(rg, nx, ny) && rg->passedBy[nx * rg->w + ny] == 0)
        {
          if (distance(rg->im.at<Vec3b>(x, y), rg->im.at<Vec3b>(nx, ny)) < var)
          {
            rg->passedBy[nx * rg->w + ny] = regionNum;
            push(&rg->stack, nx, ny); // add neighbor to stack
          }
        }
      }
    }
  }
}

/**
 * @brief Apply region growing algorithm to image
 *
 * @param rg Pointer to a RegionGrow object
 *
 * This function applies the region growing algorithm to the image stored in the
 * RegionGrow object. It initializes the currentRegion to 0 and iterates over
 * each pixel in the image. For each unprocessed pixel, it sets the currentRegion
 * to the next region number, sets the passedBy value of the pixel to the currentRegion
 * number, pushes the pixel to the stack, and calls the BFS function. After iterating
 * over all pixels, it sets the colors of each pixel based on its passedBy value,
 * displays the segmented image, and waits for a key press.
 */
void ApplyRegionGrow(RegionGrow *rg)
{
  for (int x0 = 0; x0 < rg->h; x0++)
  {
    for (int y0 = 0; y0 < rg->w; y0++)
    {
      if (rg->passedBy[x0 * rg->w + y0] == 0)
      {
        rg->currentRegion++;
        rg->passedBy[x0 * rg->w + y0] = rg->currentRegion;
        push(&rg->stack, x0, y0);
        BFS(rg, x0, y0);
      }
    }
  }

  for (int i = 0; i < rg->h; i++)
  {
    for (int j = 0; j < rg->w; j++)
    {
      double val = rg->passedBy[i * rg->w + j];
      rg->SEGS.at<Vec3b>(i, j) = val == 0 ? Vec3b(255, 255, 255) : Vec3b(val * 35, val * 90, val * 30);
    }
  }
  // save segmented image with high quality >=1000 dpi
  imwrite("../images/segmented.jpg", rg->SEGS, {IMWRITE_JPEG_QUALITY, 1000});
  imshow("Region Growing", rg->SEGS);
  waitKey(0);
}

/**
 * @brief Main function that applies the region growing algorithm to an image with a given threshold.
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 *
 * @return 0 upon successful completion
 *
 * This function checks if the number of command line arguments is 3. If it is not, it prints the usage message and returns -1.
 * If the number of arguments is 3, it initializes a RegionGrow object with the given image path and threshold.
 * It then applies the region growing algorithm to the image, frees the memory allocated for the RegionGrow object, and returns 0.
 */
int main(int argc, char **argv)
{
  if (argc != 3)
  {
    printf("Usage: %s <image_path> <threshold>\n", argv[0]);
    return -1;
  }

  RegionGrow rg;
  initRegionGrow(&rg, argv[1], atof(argv[2]));
  ApplyRegionGrow(&rg);
  freeRegionGrow(&rg);

  return 0;
}
