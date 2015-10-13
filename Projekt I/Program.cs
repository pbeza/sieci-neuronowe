using System;

namespace sieci_neuronowe
{
    class Program
    {
        static void Main(string[] args)
        {
            var parser = new Parser(args);
            if (!parser.Valid) return;
            var nn = new NeuralNetwork(parser.InputFilePath, "../../data/classification/data.test.csv", "out.txt");
            Console.ReadKey();
        }
    }
}