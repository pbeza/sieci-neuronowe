namespace gui.geo
{
    #region

    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Windows.Media;
    using System.Windows.Media.Imaging;

    #endregion

    public static class BitmapHelper
    {
        public static BitmapSource FromTypeArray(TerrainType[,] arr)
        {
            int resolutionX = arr.GetLength(0);
            int resolutionY = arr.GetLength(1);
            byte[] byteArray = new byte[arr.Length * 3];
            for (int y = 0; y < resolutionY; y++)
            {
                for (int x = 0; x < resolutionX; x++)
                {
                    if (arr[x, y] != TerrainType.Nothing)
                    {
                        int index = ((y * resolutionX) + x) * 3;
                        byteArray[index] = 0;
                        byteArray[index + 1] = 255;
                        byteArray[index + 2] = 0;
                    }
                }
            }

            return BitmapSource.Create(
                resolutionX,
                resolutionY,
                96,
                96,
                PixelFormats.Rgb24, 
                null,
                byteArray,
                resolutionX * 3);
        }
    }
}