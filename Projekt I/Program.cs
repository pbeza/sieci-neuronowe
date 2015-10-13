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
    using System.IO;

    static class Program
    {
        private const string LogFilePath = "out.txt";

        private const string TrainFilePath = ".\\data\\classification\\data.train.csv";

        private const string TestingFilePath = ".\\data\\classification\\data.test.csv";

        public static void Main(string[] args)
        {
            Console.WriteLine("> Starting application.");
            if (!File.Exists(TrainFilePath) || !File.Exists(TestingFilePath))
            {
                Console.WriteLine("No data files!");
                return;
            }

            // Nie potrzeba parsera, może na koniec
            // var parser = new Parser(args);
            // var nn = new NeuralNetwork(parser.InputFilePath, parser.TestingFilePath, LogFilePath);
            var nn = new NeuralNetwork(TrainFilePath, TestingFilePath, LogFilePath);
            nn.Run();
            Console.WriteLine("> Exiting application. Bye!");
        }
    }
}