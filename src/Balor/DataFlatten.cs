using System;

namespace Balor
{
    /// <summary>
    /// This class converts between 3D and 1D arrays. The most common use case would 
    /// be to convert the 3D images used in convolutional neural networks into a 1D 
    /// vector for feedforward neural network. Note that due to performance issues, 
    /// it currently does not do any bounds checking on input/output arrays.
    /// </summary>
    [Serializable]
    public class DataFlatten
    {
        int WIDTH;
        int HEIGHT;
        int DEPTH;
        double[,,] input;
        double[] output;

        /// <summary>
        /// Initializes a new instance of the <see cref="Cyclops.Core.DataFlatten"/> 
        /// class with the given dimensions.
        /// </summary>
        /// <param name="WIDTH">The size of the first dimension.</param>
        /// <param name="HEIGHT">The size of the second dimension.</param>
        /// <param name="DEPTH">The size of the third dimension.</param>
        public DataFlatten(int WIDTH, int HEIGHT, int DEPTH)
        {
            this.WIDTH = WIDTH;
            this.HEIGHT = HEIGHT;
            this.DEPTH = DEPTH;
            input = new double[WIDTH, HEIGHT, DEPTH];
            output = new double[WIDTH * HEIGHT * DEPTH];
        }

        /// <summary>
        /// Flattens the 3d array without doing any bounds checking.
        /// </summary>
        /// <param name="input">The 3d array.</param>
        public double[] feed(double[,,] input)
        {
            this.input = input;
            for (int x = 0; x < WIDTH; x++)
                for (int y = 0; y < HEIGHT; y++)
                    for (int z = 0; z < DEPTH; z++)
                        output[x + WIDTH * (y + DEPTH * z)] = input[x, y, z];
            return output;
        }

        /// <summary>
        /// Inflates the 1d array of length WIDTH * DEPTH * HEIGHT.
        /// </summary>
        /// <param name="error">The 1d array.</param>
        public double[,,] train(double[] error)
        {
            double[,,] _error = new double[WIDTH, HEIGHT, DEPTH];
            for (int x = 0; x < WIDTH; x++)
                for (int y = 0; y < HEIGHT; y++)
                    for (int z = 0; z < DEPTH; z++)
                        _error[x, y, z] = error[x + WIDTH * (y + DEPTH * z)];
            return _error;
        }
    }
}
