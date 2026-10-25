# 文档验证依据

正式检查采用DocumentFormat.OpenXml官方SDK的OpenXmlValidator（Office2019）：三份Word均0错误；8个结构化OMML公式、5幅嵌图、6表及引用顺序另行检查。模板Hard Gate通过，Heading 1/2/3与Body Text的段落/文字直接格式、样式文件与节设置均与原件一致。merge-runs原227、合并后227，无需修改。

工具包wml-subset.xsd自身缺少r:id声明，无法编译；business-rules.xsd只定义类型而未声明document根元素，不能对完整Word主文档直接验证。未把这两项记为通过；改用完整OpenXML SDK和模板/业务规则校验。业务检查的42个警告主要为原模板未使用的7.5pt样式及有意居中的窄表；保留指定模板样式，不将警告伪写为零。

PDF按原模板未声明纸张的设置渲染，当前正文预览为Letter。合并总论文时必须按官方要求统一纸张、版心、页码和总页数；本轮不擅改用户明确要求复制的模板节设置。

用户最新分工：不生成源程序附录Word/PDF；只提供核心代码，论文手编排。当前审计合订本12页，只含正文、模型假设、问题分析。
