# sklearn_explain

Model explanation provides the ability to interpret the effect of the predictors on the composition of an individual score. These predictors can then be ranked according to their contribution in the final score (leading to a positive or negative decision).

Model explanation has always been used in credit risk applications in presence of regulatory settings . The credit company is expected to give the customer the main (top n) reasons why the credit application was rejected (also known as reason codes).

Model explanation was also recently introduced by the European Union’s new General Data Protection Regulation (GDPR, https://arxiv.org/pdf/1606.08813.pdf) to add the possibility to control the increasing use of machine learning algorithms in routine decision-making processes.

    The law will also effectively create a “right to explanation,” whereby a user can ask for an explanation of an algorithmic decision that was made about them.
