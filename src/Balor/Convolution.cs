using System;

namespace Balor
{
    /// <summary>
    /// The Convolution class is a convolutional neural network which convolves 
    /// a set of filters over a three-dimensional input array. It implements 
    /// recitified linear units, stochastic gradient descent, and a momentum 
    /// based learning rate. Vectors should be normalized to values from 0.0 to 
    /// 1.0 before being used with this class.
    /// </summary>
    [Serializable]
    public class Convolution
    {
        const double RELU_SLOPE = 0.000001;
        const double CLIPPED_MAX = 10000.0;
        const double CLIPPED_MIN = .000001;
        const double LEARNING_RATE = 0.001;
        const double MOMENTUM_RATIO = 0.10;
        const double DROP_OUT_RATIO = 0.10;

        Random r;
        int WIDTH;
        int HEIGHT;
        int DEPTH;
        int FWIDTH;
        int FHEIGHT;
        int FDEPTH;
        double[,,] input;
        double[,,] output;
        double[,,] biases;
        double[,,,] weight;
        double[,,,] deltaw;

        /// <param name="WIDTH">The input array width.</param>
        /// <param name="HEIGHT">The input array height.</param>
        /// <param name="DEPTH">The input array depth.</param>
        /// <param name="FWIDTH">The filter width.</param>
        /// <param name="FHEIGHT">The filter height.</param>
        /// <param name="FDEPTH">The filter depth (number of filters).</param>
        public Convolution(int WIDTH, int HEIGHT, int DEPTH, int FWIDTH, int FHEIGHT, int FDEPTH)
        {
            this.WIDTH = WIDTH;
            this.HEIGHT = HEIGHT;
            this.DEPTH = DEPTH;
            this.FWIDTH = FWIDTH;
            this.FHEIGHT = FHEIGHT;
            this.FDEPTH = FDEPTH;

            r = new Random();
            input = new double[WIDTH, HEIGHT, DEPTH];
            output = new double[WIDTH, HEIGHT, FDEPTH];
            biases = new double[WIDTH, HEIGHT, FDEPTH];
            weight = new double[FWIDTH, FHEIGHT, FDEPTH, DEPTH];
            deltaw = new double[FWIDTH, FHEIGHT, FDEPTH, DEPTH];

            deltaw = new double[FWIDTH, FHEIGHT, FDEPTH, DEPTH];
            for (int fz = 0; fz < FDEPTH; fz++)
            {
                for (int x = 0; x < WIDTH; x++)
                    for (int y = 0; y < HEIGHT; y++)
                        biases[x, y, fz] = r.NextDouble() - 0.5;
                for (int fx = 0; fx < FWIDTH; fx++)
                    for (int fy = 0; fy < FHEIGHT; fy++)
                        for (int z = 0; z < DEPTH; z++)
                            weight[fx, fy, fz, z] = r.NextDouble() - 0.5;
            }
        }

        /// <param name="input">The input array of size WIDTH by HEIGHT by DEPTH.</param>
        /// <returns>The output array of size WIDTH by HEIGHT by FHEIGHT.</returns>
        public double[,,] feed(double[,,] input)
        {
            this.input = input;
            for (int x = 0; x < WIDTH; x++)
                for (int y = 0; y < HEIGHT; y++)
                    for (int fz = 0; fz < FDEPTH; fz++)
                    {
                        output[x, y, fz] = biases[x, y, fz];
                        for (int fx = 0; fx < FWIDTH; fx++)
                            for (int fy = 0; fy < FHEIGHT; fy++)
                                for (int z = 0; z < DEPTH; z++)
                                {
                                    int _x = x + fx - FWIDTH / 2;
                                    int _y = y + fy - FHEIGHT / 2;
                                    if (_x < 0 || _y < 0 || _x >= WIDTH || _y >= HEIGHT)
                                        continue;
                                    output[x, y, fz] += input[_x, _y, z] * weight[fx, fy, fz, z];
                                }
                        if (output[x, y, fz] <= 0)
                            output[x, y, fz] = RELU_SLOPE * output[x, y, fz];
                        if (output[x, y, fz] < CLIPPED_MIN)
                            output[x, y, fz] = CLIPPED_MIN;
                        if (output[x, y, fz] > CLIPPED_MAX)
                            output[x, y, fz] = CLIPPED_MAX;
                    }
            return output;
        }

        /// <param name="error">An array of error signals of size WIDTH by HEIGHT by FHEIGHT.</param>
        /// <returns>An array of error signals of size WIDTH by HEIGHT by DEPTH.</returns>
        public double[,,] train(double[,,] error)
        {
            double[,,] _error = new double[WIDTH, HEIGHT, DEPTH];
            for (int fx = 0; fx < FWIDTH; fx++)
                for (int fy = 0; fy < FHEIGHT; fy++)
                    for (int fz = 0; fz < FDEPTH; fz++)
                        for (int z = 0; z < DEPTH; z++)
                            for (int x = 0; x < WIDTH; x++)
                                for (int y = 0; y < HEIGHT; y++)
                                {
                                    int _x = x + fx - FWIDTH / 2;
                                    int _y = y + fy - FHEIGHT / 2;
                                    if (_x < 0 || _y < 0 || _x >= WIDTH || _y >= HEIGHT)
                                        continue;
                                    var deriv = error[x, y, fz];
                                    if (output[x, y, fz] <= 0)
                                        deriv *= RELU_SLOPE;
                                    if (r.NextDouble() < DROP_OUT_RATIO)
                                        biases[x, y, fz] -= LEARNING_RATE * deriv;
                                    if (r.NextDouble() < DROP_OUT_RATIO)
                                    {
                                        deltaw[fx, fy, fz, z] = deriv * input[_x, _y, z] + MOMENTUM_RATIO * deltaw[fx, fy, fz, z];
                                        weight[fx, fy, fz, z] -= LEARNING_RATE * deltaw[fx, fy, fz, z];
                                    }
                                    _error[_x, _y, z] += deriv * weight[fx, fy, fz, z];
                                }
            return _error;
        }
    }
}
