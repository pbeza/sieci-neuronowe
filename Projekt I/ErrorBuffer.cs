namespace sieci_neuronowe
{
    public class ErrorBuffer
    {
        private double errorAccum;

        private readonly double[] errorValues;

        private int errorIndexLast;

        private int errorIndexFirst;

        private int errorCount;

        private readonly int maxErrorCount;

        public ErrorBuffer(int maxCount)
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
}