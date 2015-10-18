using Encog.Engine.Network.Activation;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace sieci_neuronowe
{
    public static class NeuralNetworkFileParser
    {
        private const char TextSeparator = ' ';
        private const char CharacterStartingComment = '#';
        private const string ExceptionPostfix = "Error reading neural network file. Make sure you have correct file's format. ";
        private const bool IsBiasSet = true;

        public static BasicNetwork Parse(string path)
        {
            // Note: Below line does NOT load whole file into memory at once (which is good)

            var lines = File.ReadLines(path);
            var enumerator = lines.GetEnumerator();

            // Fetch first not empty line which should be sequence of numbers of neurons in subsequent layers

            var neuronsInLayers = GetListOfNeuronsNumberInLayers(enumerator);
            var totalLayersNumber = neuronsInLayers.Count;

            // Count number of layers which is sum of: input layer + all hidden layers + output layer

            if (totalLayersNumber < 3)
            {
                ThrowErrorFileLoadFile("At least one hidden layer expected.");
            }

            // Read and save bias for every network layer

            var layersBiases = GetListOfLayersBiases(enumerator);

            // Check if bias was specified for every layer

            if (layersBiases.Count != totalLayersNumber)
            {
                ThrowErrorFileLoadFile("Number of specified biases should be equal to number of all layers.");
            }

            // Add all layers to constructed neural network

            var network = new BasicNetwork();
            AddAllLayersToNeuralNetwork(neuronsInLayers, network);
            network.Structure.FinalizeStructure();

            // Add biases to neural network

            AddAllBiasesToNeuralNetwork(layersBiases, network);

            // Add all weights to constructed neural network

            return AddAllWeightsToConstructedNeuralNetwork(enumerator, neuronsInLayers, network);
        }

        private static BasicNetwork AddAllWeightsToConstructedNeuralNetwork(IEnumerator<string> enumerator,
                                                                            IList<int> neuronsInLayers,
                                                                            BasicNetwork network)
        {
            int processedLayers = 0, // layer no. 0 is input layer
                processedNeuronsWithinLayer = 0,
                processedNotIgnoredLines = 0,
                totalNumberOfLayers = neuronsInLayers.Count,
                numberOfNeuronsInCurrentLayer = neuronsInLayers[processedLayers],
                numberOfNeuronsInNextLayer = neuronsInLayers[processedLayers + 1];

            while (true)
            {
                var currentLine = GetFirstNotIgnoredLine(enumerator);
                if (currentLine == null)
                    break; // EOF
                processedNotIgnoredLines++;
                if (processedLayers >= totalNumberOfLayers - 1)
                {
                    var msg = string.Format("Too many neural layers. Number of processed layers is {0}. Declared number of layers is {1}.",
                                            processedLayers, totalNumberOfLayers);
                    ThrowErrorFileLoadFile(msg);
                }
                var weightsForOneVertex = GetWeightsFromLine(currentLine);
                if (weightsForOneVertex.Length != numberOfNeuronsInNextLayer)
                {
                    var msg = string.Format("Number of weights for layer no. {0} should equal {1} (is {2}).",
                                            processedLayers, numberOfNeuronsInNextLayer, weightsForOneVertex.Length);
                    ThrowErrorFileLoadFile(msg);
                }
                for (var toNeuronNumber = 0; toNeuronNumber < numberOfNeuronsInNextLayer; toNeuronNumber++)
                {
                    network.AddWeight(processedLayers, processedNeuronsWithinLayer, toNeuronNumber, weightsForOneVertex[toNeuronNumber]);
                }
                if (++processedNeuronsWithinLayer != numberOfNeuronsInCurrentLayer) continue;
                if (++processedLayers == totalNumberOfLayers - 1)
                    break;
                processedNeuronsWithinLayer = 0;
                numberOfNeuronsInCurrentLayer = neuronsInLayers[processedLayers];
                numberOfNeuronsInNextLayer = neuronsInLayers[processedLayers + 1];
            }

            // Check if file ended too early

            if (processedNotIgnoredLines != GetTotalNumberOfExpectedNotIgnoredLines(neuronsInLayers) || processedLayers != totalNumberOfLayers - 1)
                ThrowErrorFileLoadFile("Unexpected end of file. Wrong file format.");

            return network;
        }

        private static void AddAllLayersToNeuralNetwork(IEnumerable<int> neuronsInLayers, BasicNetwork network)
        {
            foreach (var neuronsInLayer in neuronsInLayers)
            {
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), IsBiasSet, neuronsInLayer));
            }
        }

        private static void AddAllBiasesToNeuralNetwork(IList<double> layersBiases, BasicNetwork network)
        {
            for (var i = 0; i < layersBiases.Count; i++)
            {
                network.SetLayerBiasActivation(i, layersBiases[i]);
            }
        }

        private static IList<int> GetListOfNeuronsNumberInLayers(IEnumerator<string> enumerator)
        {
            var currentLine = GetFirstNotIgnoredLine(enumerator);
            if (currentLine == null)
            {
                ThrowErrorFileLoadFile("File is empty.");
            }
            return GetNeuronsInLayersFromLine(currentLine);
        }

        private static IList<double> GetListOfLayersBiases(IEnumerator<string> enumerator)
        {
            var currentLine = GetFirstNotIgnoredLine(enumerator);
            if (currentLine == null)
            {
                ThrowErrorFileLoadFile("File does NOT specify layers' biases.");
            }
            return GetBiasesForEveryNetworkLayerFromLine(currentLine);
        }

        private static int GetTotalNumberOfExpectedNotIgnoredLines(IList<int> neuronsInLayers)
        {
            var totalNumberOfWeights = 0;
            for (var i = 0; i < neuronsInLayers.Count - 1; i++)
            {
                totalNumberOfWeights += neuronsInLayers[i];
            }
            return totalNumberOfWeights;
        }

        private static string GetFirstNotIgnoredLine(IEnumerator<string> enumerator)
        {
            string firstNotIgnoredLine = null;
            while (enumerator.MoveNext())
            {
                var tmp = RemoveIgnoredSubstrings(enumerator.Current);
                if (tmp == string.Empty) continue;
                if (tmp != null)
                    firstNotIgnoredLine = tmp;
                break;
            }
            return firstNotIgnoredLine;
        }

        private static string RemoveIgnoredSubstrings(string line)
        {
            var index = line.IndexOf(CharacterStartingComment);
            if (index != -1)
                line = line.Remove(index); // remove comment
            line = line.Trim();
            return Regex.Replace(line, @"\s+", " ");
        }

        private static int[] GetNeuronsInLayersFromLine(string line)
        {
            var neuronsInLayers = Array.ConvertAll(line.Split(TextSeparator), int.Parse);
            foreach (var neuronsInLayer in neuronsInLayers.Where(neuronsInLayer => neuronsInLayer <= 0))
            {
                ThrowErrorFileLoadFile("One of the specified layers not positive number of neurons.");
            }
            return neuronsInLayers;
        }

        private static double[] GetBiasesForEveryNetworkLayerFromLine(string line)
        {
            var biases = Array.ConvertAll(line.Split(TextSeparator), double.Parse);
            foreach (var bias in biases.Where(bias => bias < 0))
            {
                ThrowErrorFileLoadFile("Negative layer's bias detected.");
            }
            return biases;
        }

        private static double[] GetWeightsFromLine(string line)
        {
            var weights = Array.ConvertAll(line.Split(TextSeparator), double.Parse);
            foreach (var weight in weights.Where(weight => weight < 0))
            {
                ThrowErrorFileLoadFile("Negative neuron's weight detected.");
            }
            return weights;
        }

        private static void ThrowErrorFileLoadFile(string errorMsg)
        {
            throw new FileLoadException(ExceptionPostfix + errorMsg);
        }
    }
}
