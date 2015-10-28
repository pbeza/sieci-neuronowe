namespace sieci_neuronowe
{
    public class CircularErrorBuffer
    {
        private double errorAccum;

        private readonly double[] errorValues;


        private int errorIndexLast;

        private int errorIndexFirst;

        private int errorCount;

        private readonly int maxErrorCount;

        public CircularErrorBuffer(int maxCount)
        {
            this.maxErrorCount = maxCount;
            this.errorValues = new double[this.maxErrorCount];
        }

        public double AddError(double errorCur)
        {
            errorValues[this.errorIndexLast] = errorCur;
            this.errorIndexLast = (this.errorIndexLast + 1) % this.maxErrorCount;
            this.errorAccum += errorCur;
            if (this.errorCount == this.maxErrorCount)
            {
                this.errorAccum -= this.errorValues[this.errorIndexFirst];
                this.errorIndexFirst = (this.errorIndexFirst + 1) % this.maxErrorCount;
            }
            else
            {
                this.errorCount++;
            }

            return this.errorAccum / this.errorCount;
        }

        public double GetAverageError()
        {
            return this.errorAccum / this.errorCount;
        }
    }

    public class ErrorBuffer
    {
        private double errorAccum;

        private readonly double[] errorValues;

        private readonly bool[] isWritten;

        private int errorCount;

        public ErrorBuffer(int maxCount)
        {
            this.errorValues = new double[maxCount];
            this.isWritten = new bool[maxCount];
        }

        public double AddError(double errorCur, int index)
        {
            if (isWritten[index])
            {
                errorAccum -= errorValues[index];
            }
            else
            {
                isWritten[index] = true;
                this.errorCount++;
            }

            errorValues[index] = errorCur;
            this.errorAccum += errorCur;

            return this.errorAccum / this.errorCount;
        }

        public double GetAverageError()
        {
            return this.errorAccum / this.errorCount;
        }

        public int IndexRoulette(double getAt)
        {
            if (errorCount < errorValues.Length)
            {
                return errorCount;
            }

            double acc = 0.0;
            double threshold = errorAccum * getAt;
            for (int i = 0; i < this.errorValues.Length; i++)
            {
                acc += this.errorValues[i];
                if (acc >= threshold)
                {
                    return i;
                }
            }

            return (int)(getAt * errorValues.Length);
        }
    }
}