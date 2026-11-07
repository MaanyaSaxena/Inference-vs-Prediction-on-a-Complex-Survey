# Design-based cross-check with the survey package (strata + PSU + weights).

suppressMessages(library(survey))
options(survey.lonely.psu = "adjust")
d <- read.csv("outputs/analysis_data.csv")
des <- svydesign(ids = ~psu, strata = ~strata, weights = ~w, data = d, nest = TRUE)
xs <- setdiff(names(d), c("y_cont", "y_bin", "w", "psu", "strata"))

m_logit <- svyglm(as.formula(paste("y_bin ~", paste(xs, collapse = "+"))),
                  design = des, family = quasibinomial())
m_haz   <- svyglm(as.formula(paste("y_cont ~", paste(xs, collapse = "+"))), design = des)

tidy <- function(m) {
  s <- summary(m)$coefficients
  data.frame(term = ifelse(rownames(s) == "(Intercept)", "const", rownames(s)),
             beta = s[, 1], se = s[, 2], p = s[, 4])
}
write.csv(tidy(m_logit), "outputs/r_logit.csv", row.names = FALSE)
write.csv(tidy(m_haz),   "outputs/r_ols.csv",   row.names = FALSE)
print(svymean(~y_bin, des))   # weighted prevalence with design-based SE
cat("Design df =", degf(des), "(t-based CIs use this; small for NHANES)\n")
cat("Done. Now run: python src/crosscheck.py\n")

