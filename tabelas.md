# Documentação do Modelo de Dados

Este documento descreve o modelo de dados utilizado para armazenar informações relacionadas aos alunos, cursos, atividades, progresso, preferências de aprendizagem, sessões de estudo, recursos de aprendizagem, interações dos alunos, feedbacks da IA e histórico de perguntas e respostas da LLM.

## Entidades e Tabelas

### 1. Tabela de Alunos (`Alunos`)

Esta tabela armazena informações sobre cada aluno no sistema.

- **MatriculaAluno** (`Integer`, `primary_key=True`): Identificador único para cada aluno.
- **Nome** (`String(100)`, `nullable=False`): Nome do aluno.
- **Email** (`String(100)`, `unique=True`, `nullable=False`): Endereço de e-mail do aluno.
- **PreferenciaEstudo** (`String(50)`): Estilo de aprendizado preferido do aluno.
- **UltimoLogin** (`DateTime`): Data e hora do último login do aluno no sistema.

### 2. Tabela de Cursos (`Cursos`)

Esta tabela armazena informações sobre os cursos oferecidos.

- **IdCurso** (`Integer`, `primary_key=True`): Identificador único para cada curso.
- **NomeCurso** (`String(100)`, `nullable=False`): Nome do curso.
- **Descricao** (`String`): Descrição do curso.
- **MatriculaAluno** (`Integer`, `ForeignKey('Alunos.MatriculaAluno')`): Referência ao aluno que está associado ao curso.

### 3. Tabela de Atividades (`Atividades`)

Armazena informações sobre as atividades relacionadas a cada curso.

- **IdAtividade** (`Integer`, `primary_key=True`): Identificador único para cada atividade.
- **IdCurso** (`Integer`, `ForeignKey('Cursos.IdCurso')`): Referência ao curso relacionado.
- **Titulo** (`String(200)`, `nullable=False`): Título da atividade.
- **Descricao** (`String`): Descrição da atividade.
- **DataEntrega** (`Date`): Data de entrega da atividade.

### 4. Tabela de Progresso do Aluno (`ProgressoAluno`)

Monitora o progresso dos alunos em suas atividades.

- **IdProgresso** (`Integer`, `primary_key=True`): Identificador único para cada registro de progresso.
- **MatriculaAluno** (`Integer`, `ForeignKey('Alunos.MatriculaAluno')`): Referência ao aluno.
- **IdCurso** (`Integer`, `ForeignKey('Cursos.IdCurso')`): Referência ao curso.
- **IdAtividade** (`Integer`, `ForeignKey('Atividades.IdAtividade')`): Referência à atividade.
- **DataConclusao** (`Date`): Data de conclusão da atividade.
- **Nota** (`Float(precision=5, scale=2)`): Nota obtida pelo aluno na atividade.

### 5. Tabela de Preferências de Aprendizagem (`PreferenciasAprendizagem`)

Armazena as preferências de aprendizagem de cada aluno.

- **IdPreferencia** (`Integer`, `primary_key=True`): Identificador único para cada preferência de aprendizagem.
- **MatriculaAluno** (`Integer`, `ForeignKey('Alunos.MatriculaAluno')`): Referência ao aluno.
- **TipoPreferencia** (`String(50)`, `nullable=False`): Tipo de preferência de aprendizagem (ex.: visual, auditivo, etc.).
- **ValorPreferencia** (`String`): Detalhes adicionais sobre a preferência.

### 6. Tabela de Sessões de Estudo (`SessoesEstudo`)

Registra as sessões de estudo dos alunos.

- **IdSessao** (`Integer`, `primary_key=True`): Identificador único para cada sessão de estudo.
- **MatriculaAluno** (`Integer`, `ForeignKey('Alunos.MatriculaAluno')`): Referência ao aluno.
- **IdCurso** (`Integer`, `ForeignKey('Cursos.IdCurso')`): Referência ao curso.
- **Inicio** (`DateTime`): Início da sessão de estudo.
- **Fim** (`DateTime`): Fim da sessão de estudo.
- **Duracao** (`Interval`): Duração da sessão de estudo.
- **Produtividade** (`Integer`): Medida de produtividade da sessão de estudo.
- **FeedbackDoAluno** (`String`): Feedback fornecido pelo aluno sobre a sessão.

### 7. Tabela de Recursos de Aprendizagem (`RecursosAprendizagem`)

Contém informações sobre os recursos de aprendizagem disponíveis.

- **IdRecurso** (`Integer`, `primary_key=True`): Identificador único para cada recurso de aprendizagem.
- **IdCurso** (`Integer`, `ForeignKey('Cursos.IdCurso')`): Referência ao curso associado.
- **Titulo** (`String(200)`, `nullable=False`): Título do recurso.
- **Tipo** (`String(50)`): Tipo de recurso (ex.: livro, vídeo, etc.).
- **URL** (`String`): URL para o recurso (se aplicável).
- **Conteudo** (`String`): Descrição ou conteúdo adicional do recurso.

### 8. Tabela de Interações do Aluno (`InteracoesAluno`)

Registra as interações dos alunos com os recursos de aprendizagem.

- **IdInteracao** (`Integer`, `primary_key=True`): Identificador único para cada interação.
- **MatriculaAluno** (`Integer`, `ForeignKey('Alunos.MatriculaAluno')`): Referência ao aluno.
- **IdRecurso** (`Integer`, `ForeignKey('RecursosAprendizagem.IdRecurso')`): Referência ao recurso utilizado.
- **TipoInteracao** (`String(50)`): Tipo de interação (ex.: visualização, download, etc.).
- **DataInteracao** (`DateTime`): Data e hora da interação.
- **Duracao** (`Interval`): Duração da interação com o recurso.

### 9. Tabela de Feedback de IA (`FeedbackIA`)

Armazena feedbacks fornecidos pela IA ao aluno.

- **IdFeedback** (`Integer`, `primary_key=True`): Identificador único para cada feedback.
- **MatriculaAluno** (`Integer`, `ForeignKey('Alunos.MatriculaAluno')`): Referência ao aluno.
- **TipoFeedback** (`String(50)`): Tipo de feedback (ex.: sugestão, correção, etc.).
- **ConteudoFeedback** (`String`): Conteúdo detalhado do feedback.
- **DataCriacao** (`DateTime`, `server_default='CURRENT_TIMESTAMP'`): Data e hora da criação do feedback.

### 10. Tabela de Histórico de Perguntas e Respostas da LLM (`HistoricoPerguntasRespostasLLM`)

Registra todas as perguntas feitas pelos alunos e as respostas geradas pela LLM.

- **IdPerguntaResposta** (`Integer`, `primary_key=True`): Identificador único para cada pergunta e resposta.
- **MatriculaAluno** (`Integer`, `ForeignKey('Alunos.MatriculaAluno')`): Referência ao aluno.
- **DataHoraPergunta** (`DateTime`): Data e hora da pergunta feita pelo aluno.
- **ConteudoPergunta** (`String`): Texto da pergunta feita pelo aluno.
- **DataHoraResposta** (`DateTime`): Data e hora da resposta fornecida pela LLM.
- **ConteudoResposta** (`String`): Texto da resposta gerada pela LLM.
- **ConfidenciaResposta** (`Float`): Confiança da resposta gerada pela LLM.
- **TipoPergunta** (`String(50)`): Categoria ou tipo da pergunta (ex.: conceito, cálculo, geral, etc.).

---

Este modelo de dados permite uma análise detalhada das interações dos alunos com o sistema, permitindo personalização e melhorias contínuas com base no comportamento e nas necessidades dos alunos.
