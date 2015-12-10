using System;
using System.IO;
using System.Collections.Generic;

namespace Balor.CLI
{
    class Program
    {
        const double SPLIT = 0.7;
        const double EPOCHS = 100;

        static double[][] inputs;
        static double[][] outputs;

        /// <summary>
        /// This is the main entry point. It loads the iris data set and trains
        /// a feedforward neural network to classify the flowers.
        /// </summary>
        static void Main(string[] args)
        {
            Console.WriteLine("Welcome to Balor.NET v0.1!");
            Console.WriteLine("Loading iris data set...");
            loadDataSet();
            while (true)
            {
                Console.WriteLine("Ready. Press any key to start training...");
                Console.ReadLine();
                learnDataSet();
                Console.WriteLine();
            }
        }

        /// <summary>
        /// This method reads, randomizes, and stores the iris dataset. After 
        /// the method executes, the inputs and outputs arrays will be filled 
        /// with the iris data.
        /// </summary>
        static void loadDataSet()
        {
            List<double[]> input = new List<double[]>();
            List<double[]> output = new List<double[]>();
            using (StreamReader reader = new StreamReader(File.OpenRead("iris.csv")))
            {
                while (!reader.EndOfStream)
                {
                    string[] values = reader.ReadLine().Split(',');
                    input.Add(new double[] {
                        double.Parse(values[0]),
                        double.Parse(values[1]),
                        double.Parse(values[2]),
                        double.Parse(values[3])
                    });
                    output.Add(new double[] {
                    values[4] == "setosa" ? 1 : 0,
                    values[4] == "versicolor" ? 1 : 0,
                    values[4] == "virginica" ? 1 : 0,
                });
                }
            }

            int i = 0;
            Random rng = new Random();
            inputs = new double[input.Count][];
            outputs = new double[input.Count][];
            while (input.Count > 0)
            {
                int index = rng.Next(input.Count);
                inputs[i] = input[index];
                outputs[i] = output[index];
                input.RemoveAt(index);
                output.RemoveAt(index);
                i++;
            }

            return;
        }

        /// <summary>
        /// This function trains a two layer feedforward network of dimensions 
        /// 4-64-3 to recognize iris data. It loops until EPOCHS is reached or 
        /// the classifier achieves a perfect score. The data is divided into 
        /// training and testing data using SPLIT; data below the SPLIT is used 
        /// for training, while the rest is used for testing.
        /// </summary>
        static void learnDataSet()
        {
            int score = int.MinValue;
            int total = int.MaxValue;
            int border = (int)(inputs.Length * SPLIT);
            FeedForward l1 = new FeedForward(4, 64);
            FeedForward l2 = new FeedForward(64, 3);
            for (int epoch = 0; epoch < EPOCHS && score != total; epoch++)
            {
                score = 0;
                total = 0;

                // train
                for (int i = 0; i < border; i++)
                {
                    var result = l2.feed(l1.feed(inputs[i]));
                    var error = new double[3];
                    for (int j = 0; j < 3; j++)
                        error[j] = result[j] - outputs[i][j];
                    l1.train(l2.train(error));
                }

                // test
                for (int i = border; i < inputs.Length; i++)
                {
                    var max_j = 0;
                    var correct_j = 0;
                    var result = l2.feed(l1.feed(inputs[i]));
                    for (int j = 0; j < 3; j++)
                    {
                        if (result[j] > result[max_j])
                            max_j = j;
                        if (outputs[i][j] > outputs[i][correct_j])
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
        }
    }
}
