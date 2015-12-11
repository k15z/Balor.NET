using System;

namespace Balor.CLI
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Welcome to Balor.NET v0.1!");
            Console.WriteLine("--------------------------------------------------");
            Console.WriteLine(
                "This version of Balor.NET doesn't have a GUI... or anything else really. It's " + 
                "just a library that implements a feedforward and a convolutional neural network " + 
                "which can be used for machine learning; it should work on both .NET and Mono, and " + 
                "has been used to succesfully learn both the iris flower and MNIST handwriting dataset."
            );
            Console.WriteLine("--------------------------------------------------");
            Console.WriteLine(
                "Check out https://github.com/k15z/balor.net for more info!"
            );
            Console.ReadLine();
        }
    }
}
