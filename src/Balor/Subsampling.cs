using System;

namespace Balor
{
    /// <summary>
    /// The Subsampling class subsamples the 3d array by performing max-pooling 
    /// in the xy-direction. If the width/height is not evenly divided by the 
    /// sample window width/height, the left over values are ignored. This is 
    /// still a work in progress and will undergo significant changes.
    /// </summary>
    [Serializable]
    public class Subsampling
    {
        int WIDTH;
        int HEIGHT;
        int DEPTH;
        int SWIDTH;
        int SHEIGHT;
        double[,,] input;
        double[,,] output;

        /// <param name="WIDTH">The input width.</param>
        /// <param name="HEIGHT">The input height.</param>
        /// <param name="DEPTH">The input depth.</param>
        /// <param name="SWIDTH">The sampling window width.</param>
        /// <param name="SHEIGHT">The sampling window height.</param>
        public Subsampling(int WIDTH, int HEIGHT, int DEPTH, int SWIDTH, int SHEIGHT)
        {
            this.WIDTH = WIDTH;
            this.HEIGHT = HEIGHT;
            this.DEPTH = DEPTH;
            this.SWIDTH = SWIDTH;
            this.SHEIGHT = SHEIGHT;
            input = new double[WIDTH, HEIGHT, DEPTH];
            output = new double[WIDTH / SWIDTH, HEIGHT / SHEIGHT, DEPTH];
        }

        /// <param name="input">An array of size WIDTH by HEIGHT by DEPTH</param>
        /// <returns>An array of size WIDTH/SWIDTH by HEIGHT/SHEIGHT by DEPTH.</returns>
        public double[,,] feed(double[,,] input)
        {
            this.input = input;
            for (int x = 0; x < WIDTH / SWIDTH; x++)
                for (int y = 0; y < HEIGHT / SHEIGHT; y++)
                    for (int z = 0; z < DEPTH; z++)
                    {
                        output[x, y, z] = double.MinValue;
                        for (int sx = 0; sx < SWIDTH; sx++)
                            for (int sy = 0; sy < SHEIGHT; sy++)
                            {
                                int _x = x * SWIDTH + sx;
                                int _y = y * SHEIGHT + sy;
                                if (output[x, y, z] < input[_x, _y, z])
                                    output[x, y, z] = input[_x, _y, z];
                            }
                    }
            return output;
        }

        /// <param name="error">An array of size WIDTH/SWIDTH by HEIGHT/SHEIGHT by DEPTH.</param>
        /// <returns>>An array of size WIDTH by HEIGHT by DEPTH</returns>
        public double[,,] train(double[,,] error)
        {
            double[,,] _error = new double[WIDTH, HEIGHT, DEPTH];
            for (int x = 0; x < WIDTH / SWIDTH; x++)
                for (int y = 0; y < HEIGHT / SHEIGHT; y++)
                    for (int z = 0; z < DEPTH; z++)
                    {
                        int best_x = 0, best_y = 0;
                        double max_value = double.MinValue;
                        for (int sx = 0; sx < SWIDTH; sx++)
                            for (int sy = 0; sy < SHEIGHT; sy++)
                            {
                                int _x = x * SWIDTH + sx;
                                int _y = y * SHEIGHT + sy;
                                if (max_value < input[_x, _y, z])
                                {
                                    best_x = _x;
                                    best_y = _y;
                                    max_value = input[_x, _y, z];
                                }
                            }
                        _error[best_x, best_y, z] = error[x, y, z];
                    }
            return _error;
        }
    }
}
