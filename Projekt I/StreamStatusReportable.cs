namespace sieci_neuronowe
{
    using System.IO;

    using Encog;

    public class StreamStatusReportable : IStatusReportable
    {
        #region Fields

        private readonly StreamWriter writter;

        #endregion

        #region Constructors and Destructors

        public StreamStatusReportable(StreamWriter writter)
        {
            this.writter = writter;
        }

        #endregion

        #region Public Methods and Operators

        public void Report(int total, int current, string message)
        {
            if (total == 0)
            {
                this.writter.WriteLine(current + " : " + message);
            }
            else
            {
                this.writter.WriteLine(current + "/" + total + " : " + message);
            }
        }

        #endregion
    }
}