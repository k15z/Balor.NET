using System;

namespace Balor
{
    /// <summary>
    /// The FeedForward class is a fully-connected feedforward neural network 
    /// which implements recitified linear units, stochastic gradient descent, 
    /// and a momentum-based learning rate. Vectors should be normalized to 
    /// values from 0.0 to 1.0 before being used with this class.
    /// </summary>
    /// <example>
    /// This sample shows how to stack two FeedForward objects on top of each 
    /// other to create a two-layer neural network. The 4-10-1 network below 
    /// was trained on the iris data set and achieved a 94% accuracy rate.
    /// <code>
    /// 	FeedForward l1 = new FeedForward (4, 10);
    /// 	FeedForward l2 = new FeedForward (10, 1);
    /// 	double[] result = l2.feed(l1.feed(input))
    /// 	double[] error = expected - result;
    /// 	l1.train(l2.train(error))
    /// </code>
    /// </example>
    [Serializable]
    public class FeedForward
    {
        const double RELU_SLOPE = 0.000001;
        const double CLIPPED_MAX = 10000.0;
        const double CLIPPED_MIN = .000001;
        const double LEARNING_RATE = 0.001;
        const double MOMENTUM_RATIO = 0.10;
        const double DROP_OUT_RATIO = 0.10;

        Random r;
        int INPUT;
        int OUTPUT;
        double[] input;
        double[] output;
        double[] biases;
        double[,] weight;
        double[,] deltaw;

        /// <summary>
        /// Create a new instance of the <see cref="Cyclops.Core.FeedForward"/> 
        /// class. Initialize the bias and weight arrays to small random values 
        /// in the -0.5 to 0.5 range.
        /// </summary>
        /// <param name="INPUT">The number of input nodes.</param>
        /// <param name="OUTPUT">The number of output nodes.</param>
        public FeedForward(int INPUT, int OUTPUT)
        {
            this.INPUT = INPUT;
            this.OUTPUT = OUTPUT;

            r = new Random();
            input = new double[INPUT];
            output = new double[OUTPUT];
            biases = new double[OUTPUT];
            weight = new double[INPUT, OUTPUT];
            deltaw = new double[INPUT, OUTPUT];

            for (int j = 0; j < OUTPUT; j++)
            {
                biases[j] = r.NextDouble() - 0.5;
                for (int i = 0; i < INPUT; i++)
                    weight[i, j] = r.NextDouble() - 0.5;
            }
        }

        /// <summary>
        /// Feed the input vector through the network. Compute the dot product 
        /// of the input and weight matrix, applies the ReLU function, and 
        /// clips the output values to avoid NaN issues.
        /// </summary>
        /// <param name="input">The input vector.</param>
        /// <returns>The output vector.</returns>
        public double[] feed(double[] input)
        {
            this.input = input;
            for (int j = 0; j < OUTPUT; j++)
            {
                output[j] = biases[j];
                for (int i = 0; i < INPUT; i++)
                    output[j] += weight[i, j] * input[i];
                if (output[j] <= 0)
                    output[j] = RELU_SLOPE * output[j];
                if (output[j] < CLIPPED_MIN)
                    output[j] = CLIPPED_MIN;
                if (output[j] > CLIPPED_MAX)
                    output[j] = CLIPPED_MAX;
            }
            return output;
        }

        /// <summary>
        /// Train the network using the given error signals. The error signals 
        /// should be the partial derivative of the error with respect to the 
        /// output. It computes and returns error signals that can be used to 
        /// train the preceding layer.
        /// </summary>
        /// <param name="error">The error signals for this layer.</param>
        /// <returns>The error signals for the preceding layer.</returns>
        public double[] train(double[] error)
        {
            double[] _error = new double[INPUT];
            for (int j = 0; j < OUTPUT; j++)
            {
                double deriv = error[j];
                if (output[j] <= 0)
                    deriv *= RELU_SLOPE;
                if (r.NextDouble() < DROP_OUT_RATIO)
                    biases[j] -= LEARNING_RATE * deriv;
                for (int i = 0; i < INPUT; i++)
                {
                    if (r.NextDouble() < DROP_OUT_RATIO)
                    {
                        deltaw[i, j] = deriv * input[i] + MOMENTUM_RATIO * deltaw[i, j];
                        weight[i, j] -= LEARNING_RATE * deltaw[i, j];
                    }
                    _error[i] += deriv * weight[i, j];
                }
            }
            return _error;
        }
    }
}
