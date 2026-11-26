/** Bilangan bulat dengan pemisah ribuan koma (mis. 120,560,400), seperti aplikasi Streamlit. */
export function fmtInt(n: number): string {
  return n.toLocaleString('en-US')
}

/** "59.52 s" -> ["59.52", "s"]; "6.44×" -> ["6.44", "×"]; teks lain dikembalikan utuh. */
export function splitValue(text: string): [string, string] {
  const m = /^([\d.,]+)\s*(\S*)$/.exec(text.trim())
  return m ? [m[1], m[2]] : [text, '']
}
