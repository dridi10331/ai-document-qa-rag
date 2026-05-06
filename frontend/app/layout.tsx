import "./globals.css";
import { Fraunces, Space_Grotesk } from "next/font/google";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-sans"
});

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-serif"
});

export const metadata = {
  title: "RAG Studio",
  description: "Document Q&A with hybrid retrieval and citations"
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${spaceGrotesk.variable} ${fraunces.variable}`}>
      <body>
        <div className="page-root">{children}</div>
      </body>
    </html>
  );
}
