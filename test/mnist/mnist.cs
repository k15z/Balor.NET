using System;
using System.IO;
using System.Collections.Generic;

namespace Balor.CLI
{
    class Program
    {
        const double SPLIT = 0.7;
        const double EPOCHS = 100;

        static double[][,,] input;
        static double[][] output;

        static void Main(string[] args)
        {
            Console.WriteLine("Loading MINST data set...");
            loadDataSet();
            Console.WriteLine("Ready! Learning MINST data set...");
            learnDataSet();
        }

        /// <summary>
        /// This method reads the MINST data set and loads it into memory. The 
        /// MINST data is just small enough to be completely stored in memory.
        /// </summary>
        static void loadDataSet()
        {
            List<double[,,]> inputList = new List<double[,,]>();
            List<double[]> outputList = new List<double[]>();
            using (StreamReader reader = new StreamReader(File.OpenRead("mnist.csv")))
            {
                while (!reader.EndOfStream)
                {
                    string[] values = reader.ReadLine().Split(',');

                    double[] output = new double[10];
                    output[int.Parse(values[0])] = 1.0;
                    outputList.Add(output);

                    int inp = 1;
                    double[,,] input = new double[28, 28, 1];
                    for (int x = 0; x < 28; x++)
                        for (int y = 0; y < 28; y++)
                            input[x, y, 0] = float.Parse(values[inp++]) / 255.0;
                    inputList.Add(input);
                }
            }

            int i = 0;
            Random rng = new Random();
            input = new double[inputList.Count][,,];
            output = new double[inputList.Count][];
            while (inputList.Count > 0)
            {
                int index = rng.Next(inputList.Count);
                input[i] = inputList[index];
                output[i] = outputList[index];
                inputList.RemoveAt(index);
                outputList.RemoveAt(index);
                i++;
            }
            return;
        }

        /// <summary>
        /// This method trains a convolutional neural network to recognize the 
        /// MNIST data set.
        /// </summary>
        static void learnDataSet()
        {
            int score = int.MinValue;
            int total = int.MaxValue;
            int border = (int)(input.Length * SPLIT);
            Convolution l1 = new Convolution(28, 28, 1, 5, 5, 5);
            Subsampling l2 = new Subsampling(28, 28, 5, 2, 2);
            DataFlatten l3 = new DataFlatten(14, 14, 5);
            FeedForward l4 = new FeedForward(14 * 14 * 5, 10);

            for (int epoch = 0; epoch < EPOCHS && score != total; epoch++)
            {
                score = 0;
                total = 0;

                // train
                Console.Write("Progress: ");
                for (int i = 0; i < border; i++)
                {
                    var result = l4.feed(l3.feed(l2.feed(l1.feed(input[i]))));
                    var error = new double[10];
                    for (int j = 0; j < 10; j++)
                        error[j] = result[j] - output[i][j];
                    l1.train(l2.train(l3.train(l4.train(error))));
                    if (i % 1000 == 0)
                        Console.Write(Math.Round(100.0 * i / border) + "% ");
                }
                Console.WriteLine();

                // test
                for (int i = border; i < input.Length; i++)
                {
                    var max_j = 0;
                    var correct_j = 0;
                    var result = l4.feed(l3.feed(l2.feed(l1.feed(input[0]))));
                    for (int j = 0; j < 3; j++)
                    {
                        if (result[j] > result[max_j])
                            max_j = j;
                        if (output[i][j] > output[i][correct_j])
                            correct_j = j;
                    }
                    if (max_j == correct_j)
                        score++;
                    total++;
                }
                Console.WriteLine(
                    "Epoch " + epoch + ": " +
                    Math.Round(100.0 * score / total, 1) + "%, " +
                    score + " of " + total + " correct."
                );
            }

            Console.WriteLine("Done!");
            Console.ReadLine();
        }
    }
}
