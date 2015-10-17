/* 
 * Neural network implementation for classification and regression problem.
 * This project extensively uses Encog library.
 */
namespace sieci_neuronowe
{
    using System;

    public static class Program
    {
        private const bool IfDefaultArgs = true;

        public static void Main(string[] args)
        {
            Console.WriteLine("> Starting application.");
            var parser = new Parser(IfDefaultArgs ? Parser.DefaultArgs : args);
            if (parser.InputValid)
            {
                if (!parser.ShowHelpRequested)
                {
                    var nn = new NeuralNetwork(parser);
                    nn.Run();
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
            Console.WriteLine("> Exiting application. Bye!");
        }
    }
}