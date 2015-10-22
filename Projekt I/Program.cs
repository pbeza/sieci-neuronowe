/* 
 * Neural network implementation for classification and regression problem.
 * This project extensively uses Encog library.
 */
using System;

namespace sieci_neuronowe
{
    public static class Program
    {
        public static void Run(string[] args)
        {
            var argsParser = new ArgsParser(args.Length == 0 ? ArgsParser.DefaultArgs : args);
            if (argsParser.InputValid)
            {
                if (!argsParser.ShowHelpRequested)
                {
                    var fileParser = new NeuralNetworkFileParser(argsParser.NeuralNetworkDefinitionFilePath);
                    var parsedNeuralNetwork = fileParser.Parse();
                    var neuralNetwork = new NeuralNetwork(argsParser, parsedNeuralNetwork);
                    neuralNetwork.Run();
                }
                else
                {
                    argsParser.PrintUsage(args);
                }
            }
            else
            {
                Console.WriteLine(argsParser.MessageForUser);
            }
        }

        public static void Main(string[] args)
        {
            try
            {
                Run(args);
            }
            catch (Exception e)
            {
                Console.WriteLine("Ooops. Something gone wrong. ;-(");
                Console.WriteLine();
                Console.WriteLine("Details:");
                Console.WriteLine(e.Message);
            }
        }
    }
}