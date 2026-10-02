// Load this configuration before the pinned MathJax component in mkdocs.yml.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*",
    // MkDocs copies math delimiters into table-of-contents link labels.
    processHtmlClass: "arithmatex|md-ellipsis",
  },
};
