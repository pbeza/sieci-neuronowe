namespace sieci_neuronowe
{
    using System;

    using Encog.ML.Data;
    using Encog.Neural.Error;

    public class SquareErrorFunction : IErrorFunction
    {
        public void CalculateError(IMLData ideal, double[] actual, double[] error)
        {
            for (int i = 0; i < actual.Length; i++)
            {
                var linearError = ideal[i] - actual[i];
                error[i] = linearError * Math.Abs(linearError);
            }

        }
    }
}