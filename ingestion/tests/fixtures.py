"""Shared document fixtures for parser/chunker/db tests."""

TENK_HTML = """<!DOCTYPE html>
<html>
<head><title>FORM 10-K</title><style>.x { color: red; }</style></head>
<body>
<ix:header><ix:hidden><span>HIDDENXBRL-METADATA</span></ix:hidden></ix:header>
<div style="display:none">INVISIBLE-TAGGING-JUNK</div>
<div><span>Item</span> <span>1.</span> <span>Business</span></div>
<p>Acme Widgets, Inc. designs and manufactures widgets. Revenue totaled
$<ix:nonfraction name="us-gaap:Revenues" contextref="c1">1,234</ix:nonfraction>
million in fiscal 2025.</p>
<p>First paragraph line.<br/>Second line after a break.</p>
<p>47</p>
<div>Table of Contents</div>
<div style="font-weight:bold">Item 1A. Risk Factors</div>
<p>Our business depends on widget demand, which is cyclical and difficult to
forecast. A downturn could materially harm operating results.</p>
<table>
<tr><td>Revenue</td><td>1,234</td></tr>
<tr><td>Net income</td><td>567</td></tr>
</table>
<script>console.log("ignore me");</script>
</body>
</html>
"""

FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0508</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>2026-06-08</periodOfReport>
  <issuer>
    <issuerCik>0001004980</issuerCik>
    <issuerName>PG&amp;E Corp</issuerName>
    <issuerTradingSymbol>PCG</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001234567</rptOwnerCik>
      <rptOwnerName>DOE JANE</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <officerTitle>EVP, General Counsel</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-06-08</value></transactionDate>
      <transactionCoding>
        <transactionFormType>4</transactionFormType>
        <transactionCode>S</transactionCode>
        <equitySwapInvolved>0</equitySwapInvolved>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1500</value></transactionShares>
        <transactionPricePerShare><value>12.34</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>50000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
      <ownershipNature>
        <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
      </ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <footnotes>
    <footnote id="F1">Sale executed pursuant to a Rule 10b5-1 trading plan.</footnote>
  </footnotes>
</ownershipDocument>
"""
