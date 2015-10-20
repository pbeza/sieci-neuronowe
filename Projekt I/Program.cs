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
            var cmdLinePrser = new CommandLineParser(args.Length == 0 ? CommandLineParser.DefaultArgs : args);
            if (cmdLinePrser.InputValid)
            {
                if (!cmdLinePrser.ShowHelpRequested)
                {
                    var fileParser = new NeuralNetworkFileParser(cmdLinePrser.NeuralNetworkDefinitionFilePath);
                    var parsedNeuralNetwork = fileParser.Parse();
                    var neuralNetwork = new NeuralNetwork(cmdLinePrser, parsedNeuralNetwork);
                    neuralNetwork.Run();
                }
                else
                {
                    cmdLinePrser.PrintUsage(args);
                }
            }
            else
            {
                Console.WriteLine(cmdLinePrser.MessageForUser);
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