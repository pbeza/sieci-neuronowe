/* 
 * Neural network implementation for classification and regression problem.
 * This project extensively uses Encog library.
 * 
 * To run application from command line run:
 * 
 *     ./appname -c -t ../../data/classification/data.test.csv ../../data/classification/data.train.csv
 */
using System;

namespace sieci_neuronowe
{
    class Program
    {
        private const string LogFilePath = "out.txt";

        static void Main(string[] args)
        {
            Console.WriteLine("> Starting application.");
            var parser = new Parser(args);
            var nn = new NeuralNetwork(parser.InputFilePath, parser.TestingFilePath, LogFilePath);
            nn.Run();
            Console.WriteLine("> Exiting application. Bye!");
        }
    }
}