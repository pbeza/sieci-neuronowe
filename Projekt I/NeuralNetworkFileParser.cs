using Encog.Engine.Network.Activation;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;

namespace sieci_neuronowe
{
    public static class NeuralNetworkFileParser
    {
        public const char TextSeparator = ' ';
        public const char CharacterStartingComment = '#';

        public static BasicNetwork Parse(string path)
        {
            // Note: Below line does NOT load whole file into memory at once (which is good)

            var lines = File.ReadLines(path);
            var enumerator = lines.GetEnumerator();

            // Fetch first not empty line which should be sequence of numbers of neurons in subsequent layers

            var neuronsInLayers = GetListOfNeuronsNumberInLayers(enumerator);

            // Count number of layers which is sum of: input layer + all hidden layers + output layer

            var totalNumberOfLayers = GetTotalNumberOfLayers(neuronsInLayers);

            // Add all layers to constructed neural network

            var network = new BasicNetwork();
            AddAllLayersToNetwork(neuronsInLayers, network);
            network.Structure.FinalizeStructure();

            // Add all weights to constructed neural network

            return AddAllWeightsToConstructedNeuralNetwork(enumerator, totalNumberOfLayers, neuronsInLayers, network);
        }

        private static BasicNetwork AddAllWeightsToConstructedNeuralNetwork(IEnumerator<string> enumerator, int totalNumberOfLayers,
            IList<int> neuronsInLayers, BasicNetwork network)
        {
            int processedLayers = 0, // layer #0 is input layer
                processedNeuronsWithinLayer = 0,
                processedNotIgnoredLines = 0;

            var numberOfNeuronsInCurrentLayer = neuronsInLayers[processedLayers];
            var numberOfNeuronsInNextLayer = neuronsInLayers[processedLayers + 1];

            while (true)
            {
                var currentLine = GetFirstLine(enumerator);
                if (currentLine == null)
                    break; // EOF
                processedNotIgnoredLines++;
                if (processedLayers >= totalNumberOfLayers - 1)
                    throw new FileLoadException("Error reading neural network file. Too many neural layers.");
                var weightsForOneVertex = GetWeightsFromLine(currentLine);
                if (weightsForOneVertex.Length != numberOfNeuronsInNextLayer)
                {
                    var msg = string.Format("Error reading neural network file. Number of weights for layer no. {0} should equal {1} (is {2}).",
                                            processedLayers, numberOfNeuronsInNextLayer, weightsForOneVertex.Length);
                    throw new FileLoadException(msg);
                }
                for (var toNeuronNumber = 0; toNeuronNumber < numberOfNeuronsInNextLayer; toNeuronNumber++)
                {
                    var weight = weightsForOneVertex[toNeuronNumber];
                    network.AddWeight(processedLayers, processedNeuronsWithinLayer, toNeuronNumber, weight);
                }
                if (++processedNeuronsWithinLayer != numberOfNeuronsInCurrentLayer) continue;
                if (++processedLayers == totalNumberOfLayers - 1)
                    break;
                processedNeuronsWithinLayer = 0;
                numberOfNeuronsInCurrentLayer = neuronsInLayers[processedLayers];
                numberOfNeuronsInNextLayer = neuronsInLayers[processedLayers + 1];
            }

            // Check if file is empty when we finished reading data

            if (processedNotIgnoredLines == GetTotalNumberOfWeights(neuronsInLayers) && processedLayers == totalNumberOfLayers - 1)
                return network;
            throw new FileLoadException("Error reading neural network file. Unexpected end of file. Wrong file format.");
        }

        private static void AddAllLayersToNetwork(IEnumerable<int> neuronsInLayers, BasicNetwork network)
        {
            foreach (var neuronsInLayer in neuronsInLayers)
            {
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, neuronsInLayer)); // TODO Add bias support
            }
        }

        private static int GetTotalNumberOfLayers(ICollection<int> neuronsInLayers)
        {
            var totalNumberOfLayers = neuronsInLayers.Count;
            if (totalNumberOfLayers >= 3) return totalNumberOfLayers;
            const string msg = "Error reading neural network file. At least one hidden layer expected.";
            throw new FileLoadException(msg);
        }

        private static IList<int> GetListOfNeuronsNumberInLayers(IEnumerator<string> enumerator)
        {
            var currentLine = GetFirstLine(enumerator);
            if (currentLine == null)
            {
                throw new FileLoadException("Error reading neural network file. Given file is empty.");
            }
            return GetNeuronsInLayersFromLine(currentLine);
        }

        private static int GetTotalNumberOfWeights(IList<int> neuronsInLayers)
        {
            var totalNumberOfWeights = 0;
            for (var i = 0; i < neuronsInLayers.Count - 1; i++)
            {
                totalNumberOfWeights += neuronsInLayers[i];
            }
            return totalNumberOfWeights;
        }

        private static string GetFirstLine(IEnumerator<string> enumerator)
        {
            string firstNotEmptyLine = null;
            while (enumerator.MoveNext())
            {
                var tmp = RemoveUnnecessaryChars(enumerator.Current);
                if (tmp == string.Empty) continue;
                if (tmp != null)
                    firstNotEmptyLine = (string)tmp.Clone();
                break;
            }
            return firstNotEmptyLine;
        }

        private static int[] GetNeuronsInLayersFromLine(string line)
        {
            return Array.ConvertAll(line.Split(TextSeparator), int.Parse);
        }

        private static double[] GetWeightsFromLine(string line)
        {
            return Array.ConvertAll(line.Split(TextSeparator), double.Parse);
        }
        private static string RemoveUnnecessaryChars(string line)
        {
            var index = line.IndexOf(CharacterStartingComment);
            if (index != -1)
                line = line.Remove(index); // remove comment
            line = line.Trim();
            line = Regex.Replace(line, @"\s+", " ");
            return line;
        }
    }
}
