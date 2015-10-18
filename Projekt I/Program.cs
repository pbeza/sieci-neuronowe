/* 
 * Neural network implementation for classification and regression problem.
 * This project extensively uses Encog library.
 */
using System;

namespace sieci_neuronowe
{
    public static class Program
    {
        private const bool IfDefaultArgs = true;

        public static void Run(string[] args)
        {
            var parser = new CommandLineParser(IfDefaultArgs ? CommandLineParser.DefaultArgs : args);
            if (parser.InputValid)
            {
                if (!parser.ShowHelpRequested)
                {
                    var networkFromFile = NeuralNetworkFileParser.Parse(parser.NeuralNetworkDefinitionFilePath);
                    var nn = new NeuralNetwork(parser);
                    //nn.Run();
                }
                else
                {
                    parser.PrintUsage(args);
                }
            }
            else
            {
                Console.WriteLine(parser.MessageForUser);
                parser.PrintUsage(args);
            }
        }

        public static void Main(string[] args)
        {
            Console.WriteLine("> Starting application.");
            Console.WriteLine();

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

            Console.WriteLine();
            Console.WriteLine("> Exiting application. Bye!");
        }
    }
}